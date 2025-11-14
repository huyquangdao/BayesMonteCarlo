import torch
import torch.nn as nn
from torch.distributions import Categorical

from transformers import AutoTokenizer, AutoModel
from utils.prompt import call_llm
from config.constants import LLAMA3, QWEN, CHATGPT
from baselines.PPDPP.config import PPDPPConfigForRecommendation, PPDPPConfigForNegotiation, PPDPPConfigForEmotionalSupport, PPDPPConfigForPersuation

from base.model import Model


class PPDPPModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for Class BERT-based policy model
        :param model_config: the model configuration class
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        # create the tokenizer and the backbone pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)

        self.plm = AutoModel.from_pretrained(self.model_config.plm, cache_dir=self.model_config.cached_dir)

        # prepend special tokens to the vocabulary and resize the embedding matrix of the PLM
        self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        # only applicable for the recommendation scenario
        if self.model_config.combined_action:
            # other parameters.
            # for recommendation, for other scenarios, new code might be needed.
            self.n_classses = self.model_config.n_goals * self.model_config.n_topics
        # other scenarios such as negotiation and emotional support conversation
        else:
            self.n_classses = self.model_config.n_goals
        
        self.drop_out = nn.Dropout(p=self.model_config.dropout)
        self.out_layer = nn.Linear(self.model_config.lm_size, self.n_classses)

    def forward(self, batch):
        """
        Forward function
        :param batch: a batched tensor data
        :return:
        """
        cls_token = self.plm(**batch['context'])[0][:, 0, :]
        # appluing the dropout label
        cls_token = self.drop_out(cls_token)
        # producing the outputs
        logits = self.out_layer(cls_token)
        return logits

    def rewrite_action(self, inputs, action, **kwargs):
        dialogue = ''
        for utt in inputs:
            if isinstance(self.model_config, PPDPPConfigForRecommendation):
                if utt['role'] == "user":
                    role = "User"
                else:
                    role = "Recommender"
            elif isinstance(self.model_config, PPDPPConfigForNegotiation):
                if utt['role'] == "user":
                    role = "Seller"
                else:
                    role = "Buyer"
            elif isinstance(self.model_config, PPDPPConfigForEmotionalSupport):
                if utt['role'] == "user":
                    role = "Therapist"
                else:
                    role = "Patient"
            elif isinstance(self.model_config, PPDPPConfigForPersuation):
                if utt['role'] == "user":
                    role = "Persuadee"
                else:
                    role = "Persuadee"
            dialogue += f"{role}: {utt['content']} "
                            
        meta_prompt = self.model_config.meta_prompt                    
        prompt = [
            {"role": "system", "content": self.model_config.rewrite_prompt},
            {"role": "user", "content": self.model_config.rewrite_prompt_cot.format(meta_prompt, dialogue, action)}
        ]
        
        print("action: ", action)
        
        # calling the llm to predict the action
        responses = call_llm(prompt, temperature=0.6, 
                             max_token= 50,
                             model_type= self.model_config.model_type,
                             **kwargs
                             )
        
        # print(action)
        return responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()



class PreferencePPDPPModel(PPDPPModel):

    def __init__(self, model_config, **kwargs):
        """
        constructor for class preference-based PPDPP model
        :param model_config: the configuration of the model
        :param kwargs: other keywords parameters
        """
        super().__init__(model_config, **kwargs)

        # preference layer
        # computing a preference vector over a set of objectives
        # reward embedding
        # this layer convert a reward vector to a embedding vector for the model
        self.objective_embedding = nn.Linear(self.model_config.n_objectives,
                                             self.model_config.objective_embedding_size)

        # self.criterion = torch.nn.CrossEntropyLoss()
        # actor model
        # we use the state embedding and objective embedding
        # Actor(a|s,w)
        self.actor = nn.Linear(self.model_config.lm_size + self.model_config.objective_embedding_size,
                               self.n_classses)

        # critic model
        # Critic(s,w)
        self.critic = nn.Linear(self.model_config.lm_size + self.model_config.objective_embedding_size,
                                1)

        # the hs and fs functions are used for reward shaping
        # we instance hs and gs with MLP layer.
        self.gs = nn.Sequential(
            nn.Linear(self.model_config.lm_size, self.model_config.reward_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(self.model_config.reward_hidden_size, 1)
        )

        self.hs = nn.Sequential(
            nn.Linear(self.model_config.lm_size + self.model_config.objective_embedding_size,
                      self.model_config.reward_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(self.model_config.reward_hidden_size, 1)
        )

    def forward(self, batch, is_computing_reward=False, is_actor_critic_training=False):
        """
        Forward function
        :param batch: a batched tensor data
        :param is_computing_reward: if we are computing the reward
        :param is_actor_critic_training: if we are training the actor-critic model
        :return:
        """
        # predicting the policy
        # using the state and objective embedding
        # objective embedding
        objective_embedding = self.objective_embedding(batch['w'])

        # cls token as the state
        state = self.plm(**batch['context'])[0][:, 0, :]
        # applying the dropout layer

        feature = torch.cat([state, objective_embedding], dim=-1)
        feature = self.drop_out(feature)

        # producing the outputs
        logits = self.actor(feature)

        # if we are computing the reward signal
        if is_computing_reward:
            # assert we have the next state
            assert 'next_state' in batch

            # cls token as the state
            state = self.plm(**batch['context'])[0][:, 0, :]
            # appluing the dropout on the current feature
            # next state
            next_state = self.plm(**batch['next_state'])[0][:, 0, :]

            # compute the estimated reward function
            reward = self.compute_reward(objective_embedding, state, next_state)

            # preference training:
            if not is_actor_critic_training:
                return logits, torch.exp(reward)

            next_feature = torch.cat([next_state, objective_embedding])

            # compute the values for critic
            values = self.critic(feature)
            next_values = self.critic(next_feature)

            # compute scalarized values
            values = (batch['w'] * values).sum(dim=-1)
            next_values = (batch['w'] * next_values).sum(dim=-1)

            # return the log pi, reward, v(s) and v(s')
            return logits, torch.exp(reward), values, next_values

        return logits

    def compute_reward(self, objective_embedding, state, next_state, gamma=0.2):
        """
        function that estimate the reward for a particular state and preference
        :param w: the preference vector
        :param state: the current state
        :param next_state: the next state
        :param gamma: a hyper-parameter
        :return: a float score indicating the reward for the current turn.
        """
        # compute the estimated common reward function
        gs = self.gs(state)
        # shape = [bs, state_size] and [bs, reward_embedding_size]
        feature = torch.cat([state, objective_embedding], dim=-1)
        feature = self.drop_out(feature)
        # next feature
        next_feature = torch.cat([next_state, objective_embedding], dim=-1)
        next_feature = self.drop_out(next_feature)
        # compute the user-oriented reward part
        hs_prime = self.hs(next_feature)
        hs = self.hs(feature)
        return gs + gamma * hs_prime - hs

    def manipulate_gradient_update(self, is_preference_block=True, flag=True):
        """
        method that manipulate the gradient update of each block in the model
        :param is_preference_block: True if we wish to manipulate the gradient of the preference block
        :param flag: the status of the gradient update
        :return: None
        """
        # during lr training we freeze parameters of the plm and the objective embedding layer
        for name, parameters in self.named_parameters():
            parameters.requires_grad_(False)
        # manipulating gradient of the preference estimation block
        if is_preference_block:
            self.gs.requires_grad_(flag)
            self.hs.requires_grad_(flag)
        # manipulating the gradient update of the policy part
        # freeze or unfreeze parameters of the objective embedding, plm and out layer
        else:
            self.critic.requires_grad_(flag)
            self.actor.requires_grad_(flag)

    def compute_state_resp(self, state_dict, w):
        """
        method that compute the state representations, only used during actor-critic training
        :param state_dict: the batch data
        :return:
        """
        with torch.no_grad():
            objective_embedding = self.objective_embedding(w)
            # cls token as the state
            state = self.plm(**state_dict)[0][:, 0, :]
            # applying the dropout layer

            feature = torch.cat([state, objective_embedding], dim=-1)
            feature = self.drop_out(feature)
            return feature

    def compute_state_value(self, state_resp):
        """
        method that compute the state representations, only used during actor-critic training
        :param state_resp: the state representations
        :return:
        """
        # compute the values for critic
        values = self.critic(state_resp)
        return values.squeeze(-1)

    def compute_log_probs(self, state_resp):
        """
        method that compute the log probs
        :param state_resp: the state representations
        :return: log probs
        """
        # producing the outputs
        logits = self.actor(state_resp)
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        return dist.log_prob(action)
