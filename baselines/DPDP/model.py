from collections import defaultdict as ddict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel

from base.model import Model
from baselines.GDP_Zero.model import GDPZeroModel
from baselines.GDP_Zero.config import GDPZeroConfigForRecommendation


class DPDPQnet(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for Class BERT-based policy model
        :param model_config: the model configuration class
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        # if we predict both goal, topic at a
        # only applicable for the recommendation scenario
        if self.model_config.combined_action:
            # other parameters.
            # for recommendation, for other scenarios, new code might be needed.
            self.n_classses = self.model_config.n_goals * self.model_config.n_topics
        # other scenarios such as negotiation and emotional support conversation
        else:
            self.n_classses = self.model_config.n_goals

        self.fc_1 = nn.Linear(self.model_config.lm_size, self.model_config.hidden_size)
        self.fc_2 = nn.Linear(self.model_config.hidden_size, self.n_classses)

    def forward(self, x):
        """
        Forward function
        :param batch: a batched tensor data
        :return:
        """
        x = torch.relu(self.fc_1(x))
        # producing the outputs
        q_value = self.fc_2(x)
        return q_value



class DPDPModel(Model):
    def __init__(self, model_config, **kwargs):
        
        super().__init__(model_config, **kwargs)
        # create the tokenizer and the backbone pretrained language model
        # pretrained lm backbone
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)

        self.plm = AutoModel.from_pretrained(self.model_config.plm, cache_dir=self.model_config.cached_dir)

        # prepend special tokens to the vocabulary and resize the embedding matrix of the PLM
        self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        # dropout
        self.dropout = nn.Dropout(model_config.dropout)
        # self.act = sorted(list(act[args.data_name].keys()))
        # self.inv_act = {act: idx for idx, act in enumerate(self.act)}
        
        self.Q_head = DPDPQnet(model_config, **kwargs)
        self.Q_head_target = DPDPQnet(model_config, **kwargs)

        self.classifier = nn.Linear(model_config.lm_size, self.Q_head.n_classses)

        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=model_config.rl_learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.model_config = model_config

        # recommendation scenario
        if isinstance(self.model_config, GDPZeroConfigForRecommendation):
            # set the domain for the mcts config
            self.model_config.mcts_config.set_params({
                'domain': self.model_config
            })

        self.saved_log_probs = []
        self.saved_qvs = []
        self.rewards = []

        self.device = model_config.device
        
        self.mcts = GDPZeroModel(model_config.mcts_config)

        self.ent_bound = model_config.entropy_bound
        self.sub_value = model_config.sub_value
        self.success_base = model_config.success_base
        
        self.gamma = model_config.gamma
        self.lmbda = model_config.lmbda

        self.objective_weight = torch.Tensor(self.model_config.obj_to_weight[self.model_config.prioritized_objective]).to(self.device).unsqueeze(0)

        self.apply_policy_times = 0.0
        self.apply_mcts_times = 0.0
        self.apply_chatgpt_times = 0.0
        
        self.update_target_qnet()
        for p in self.Q_head_target.parameters():
            p.requires_grad = False
        
        self.action_freq = ddict(int)
        self.thresh_history = []

    def build_input(self, states):
        def pad_sequence(inputs, attention_masks):
            max_length = max([len(inp) for inp in inputs])
            attention_masks = [attn_mask + [0] * (max_length - len(inputs[idx])) for idx, attn_mask in enumerate(attention_masks)]
            inputs = [inpt + [self.tokenizer.pad_token_id] * (max_length - len(inpt)) for inpt in inputs]
            return inputs, attention_masks
        
        inps, attention_masks = [], []
        for state in states:
            dial_id = []
            for turn in state[::-1]:
                s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
                if len(dial_id) + len(s) > self.args.max_seq_length:
                    break
                dial_id = s[1:] + dial_id
            inp = s[:1] + dial_id
            inps.append(inp.copy())
            attention_masks.append([1] * len(inp))
        inps, attention_masks = pad_sequence(inps, attention_masks)
        return inps, attention_masks

    def forward(self, batch, is_test = False):
        states = batch['context']
        actions = batch['labels']
        rewards = batch['rewards']

        next_states = batch['next_state']
        dones = batch['done']

        # print(target_qvs)
        outputs = self.plm(**states)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # policy net
        logits = self.classifier(pooled_output)

        if not is_test:
            if actions is not None:

                # temporally use for testing
                target_qvs = rewards
                
                # scalarization
                target_qvs = (self.objective_weight * target_qvs).sum(dim = -1)
                
                # critic
                q_value = self.Q_head(pooled_output)
                qa_value = torch.gather(q_value, dim=-1, index=actions.view(-1, 1))
                loss_fct = torch.nn.MSELoss(reduction='mean')

                # Critic loss
                # Q(s,a) and MC(s,a)
                critic_loss = loss_fct(qa_value.view(-1), target_qvs.view(-1))
                
                # max_actions = logits.view(-1, len(self.act)).argmax(dim=-1)
                # for action in max_actions.detach().cpu().tolist():
                #     self.action_freq[action] += 1
                actor_probs = logits.view(-1, logits.size(-1)).softmax(dim=-1).gather(dim=1, index=actions.view(-1, 1))
                
                # advantage
                td_delta = target_qvs - qa_value

                # policy gradient
                actor_loss = torch.mean(-torch.log(actor_probs).view(-1) * td_delta.detach())
                
                # loss
                loss = actor_loss + self.model_config.critic_loss_w * critic_loss
                return loss, actor_loss.item(), critic_loss.item()
        else:
            return logits, self.Q_head(pooled_output)

    def encode_state(self, state):
        inp, attn_mask = self.build_input(state) if isinstance(state[0], list) else self.build_input([state])
        inp = torch.tensor(inp).long().to(self.device)
        attn_mask = torch.tensor(attn_mask).long().to(self.device)

        outputs = self.policy(input_ids=inp, attention_mask=attn_mask)
        pooled_output = outputs[1]
        return self.dropout(pooled_output)
    
    def apply_actor(self, state_encoding):
        logits = self.classifier(state_encoding)
        dist = nn.functional.softmax(logits, dim=1)
        return dist
    
    def apply_critic(self, state_encoding):
        qvs = self.Q_head(state_encoding)
        return qvs
    
    def apply_policy(self, state):
        pooled_output = self.encode_state(state)
        logits = self.classifier(pooled_output)
        dist = nn.functional.softmax(logits, dim=1)
        qvs = self.Q_head(pooled_output)
        return dist, qvs
    
    def select_action(self, state, mcts_state, action=None, is_test=False, transition_dict=None):
        use_mcts = True
        action_dist, qvs = self.apply_policy(state)
        m = Categorical(action_dist)
        if action is None:
            if is_test:
                # entropy = utils.safe_entropy(action_dist)
                # if entropy <= self.ent_bound:
                topk_probs, _ = torch.topk(action_dist, k=2)
                self.logger.info('action distribution: {}'.format(action_dist.detach().cpu().tolist()))
                self.logger.info('select {}th percentiles...'.format(self.args.mcts_applied_ratio * 100))
                if self.args.mcts_applied_ratio == 0.0:
                    sub_value = 0.0
                elif self.args.mcts_applied_ratio == 1.0:
                    sub_value = 1.0
                else:
                    sub_value = np.percentile(self.thresh_history, self.args.mcts_applied_ratio * 100) if len(self.thresh_history) >= 2 else self.sub_value
                if topk_probs[0][0] - topk_probs[0][1] > sub_value:     # sub_value 大于 1 则全部走 mcts，小于等于0,则全部走 policy
                    self.logger.info('max prob - second max prob = {} >= {}'.format(topk_probs[0][0] - topk_probs[0][1], sub_value))
                    action = action_dist.argmax().item()
                    reward, full_mcts_history = None, None
                    self.logger.info('Choose action "{}" by Policy Network...'.format(self.act[action]))
                    self.apply_policy_times += 1
                    use_mcts = False
                else:
                    self.logger.info('max prob - second max prob = {} < {}'.format(topk_probs[0][0] - topk_probs[0][1], sub_value))
                    mcts_state, reward, full_mcts_history, transition_dict, apply_chatgpt_times = self.select_action_by_mcts(mcts_state, state, transition_dict)
                    action = mcts_state[-2][1]                      # 使用 mcts_state 的倒数第二个记录的 strategy 作为动作
                    action = self.inv_act[action]
                    self.logger.info('Choose action "{}" by MCTS...'.format(self.act[action]))
                    self.apply_mcts_times += 1
                    self.apply_chatgpt_times += apply_chatgpt_times
                self.thresh_history.append((topk_probs[0][0] - topk_probs[0][1]).item())
            else:
                # action = m.sample()
                mcts_state, reward, full_mcts_history, transition_dict, apply_chatgpt_times = self.select_action_by_mcts(mcts_state, state, transition_dict)
                action_str = mcts_state[-2][1]
                action = self.inv_act[action_str]
                action_tensor = torch.tensor([action]).long().to(action_dist.device)
                self.saved_log_probs.append(m.log_prob(action_tensor))
                self.saved_qvs.append(qvs.gather(1, action_tensor.unsqueeze(dim=-1)).squeeze(dim=-1))
                self.logger.info('Choose action "{}" by MCTS...'.format(self.act[action]))
                self.apply_mcts_times += 1
                self.apply_chatgpt_times += apply_chatgpt_times
        else:
            if not is_test:
                action_tensor = torch.tensor([action]).long().to(action_dist.device)
                self.saved_log_probs.append(m.log_prob(action_tensor))
                self.saved_qvs.append(qvs.gather(1, action_tensor.unsqueeze(dim=-1)).squeeze(dim=-1))
            reward, full_mcts_history = None, None
            self.logger.info('Choose action "{}" from searched successful path by MCTS...'.format(self.act[action]))
        self.action_freq[action] += 1
        return self.act[action], mcts_state, reward, full_mcts_history, transition_dict, use_mcts
    

    def optimize_model(self, transition_dict, logger):
        logger.info('Start training ...')
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards).to(self.device)
        loss_fct = nn.MSELoss(reduction='mean')
        qa_values = torch.cat(self.saved_qvs, dim=-1)
        critic_loss = loss_fct(qa_values, rewards)
        td_delta = rewards - qa_values
        log_probs = torch.cat(self.saved_log_probs, dim=-1)
        policy_loss = (-log_probs * td_delta.detach()).mean()
        loss = policy_loss + self.args.critic_loss_w * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.saved_qvs[:]
        del transition_dict
        
        return policy_loss.item(), critic_loss.item(),
    
    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
    
    def update_target_qnet(self):
        self.Q_head_target.load_state_dict(self.Q_head.state_dict())