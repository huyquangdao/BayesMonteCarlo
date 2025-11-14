import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel

from base.model import Model

from loguru import logger
from utils.game import random_weights
import torch.nn.functional as F


class QHead(Model):

    def __init__(self, n_classes, model_config, **kwargs):
        """
        constuctor for class Qhead
        :param n_classes: the number of classes
        :param model_config: the model configuration
        :param kwargs:
        """
        super().__init__(model_config, **kwargs)

        # objective embedding
        # this layer convert a preference vector to a embedding vector for the model
        self.objective_embedding = nn.Linear(self.model_config.n_objectives,
                                             self.model_config.objective_embedding_size)

        self.projector = nn.Sequential(
            nn.Linear(
                self.model_config.lm_size,
                self.model_config.mlp_hidden_size * 2
            ),
            nn.ReLU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(self.model_config.mlp_hidden_size * 2,
                      self.model_config.mlp_hidden_size
                      )
        )

        # actor network
        # policy \pi(s,w) -> A
        self.actor = nn.Sequential(
            nn.Linear(
                # phi(s), w -> n_objectives
                self.model_config.mlp_hidden_size + self.model_config.objective_embedding_size,
                self.model_config.mlp_hidden_size
                ),
            nn.ReLU(),
            nn.Linear(
                # phi(s), w -> n_objectives
                self.model_config.mlp_hidden_size,
                n_classes),
        )

        # critic network
        # Q(s,a,w) -> d * |A|
        self.critic = nn.Sequential(
            nn.Linear(
                # phi(s), w -> n_objectives
                self.model_config.mlp_hidden_size + self.model_config.objective_embedding_size,
                self.model_config.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(
                # phi(s), w -> n_objectives
                self.model_config.mlp_hidden_size,
                self.model_config.n_objectives * n_classes
            ),
        )

    def forward(self, feature, w):
        """
        forward method
        :param feature: the lm state
        :param w: the objective weight
        :return:
        """
        # preference embedding
        w_embedding = self.objective_embedding(w)

        # w_embedding = w
        # bs, n_preferences
        bs = feature.size(0)
        n_preferences = w_embedding.size(0)

        # projecting the feature to a lower dimensional space
        feature = self.projector(feature)
        if feature.size(0) != w_embedding.size(0):
            w_embedding = w_embedding.repeat(1, bs).view(-1, w_embedding.size(-1))
            feature = feature.repeat(n_preferences, 1).view(-1, feature.size(-1))

        new_feature = torch.cat([feature, w_embedding], dim=-1)
        # computing the logits using the current state
        # a ~ \pi(.|s,w)
        pi = self.actor(new_feature)
        value = self.critic(new_feature)
        return pi, value


class SetMaxPADPPModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for class MODPL model
        :param model_config: the configuration of the model
        :param kwargs: other keywords parameters
        """
        super().__init__(model_config, **kwargs)
        # create the tokenizer and the backbone pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)

        self.plm = AutoModel.from_pretrained(self.model_config.plm,
                                             cache_dir=self.model_config.cached_dir)

        # prepend special tokens to the vocabulary and resize the embedding matrix of the PLM
        self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        # freeze the parameters for the pretrained language model
        if model_config.freeze_plm:
            logger.info("Freezing the parameters of the pretrained language modek")
            for name, params in self.plm.named_parameters():
                params.requires_grad_(False)

        self.drop_out = nn.Dropout(p=self.model_config.dropout)
        # for negotiation, esc, we only have strategies i.goals
        n_classes = self.model_config.n_goals

        # if we predict both goal, topic at a time
        # i.e the recommendation scenario.
        if self.model_config.combined_action:
            # other parameters.
            # for recommendation, for other scenarios, new code might be needed.
            n_classes = self.model_config.n_goals * self.model_config.n_topics

        # Q network
        # we use the state embedding, context embedding and objective embedding
        self.Q_head = QHead(n_classes, model_config)

    def forward(self, batch, is_pretraining=True):
        """
        The forward function for class MODPL for recommendation
        :param batch
        :return:
        """
        batch_state = batch['context']
        # cls token as the state
        feature = self.plm(**batch_state).last_hidden_state[:, 0, :]

        # pretraining step
        # we perform offline multi-objective reinforcement learning
        logits, value = self.Q_head(feature, batch['w'])
        return logits, value

    def update_target_network(self):
        """
        copy the parameters of the Q to the Q target network
        :return:
        """
        self.Q_head_target.load_state_dict(self.Q_head.state_dict())

    def compute_features(self, batch):
        """
        function that estimate the reward for a particular state and preference
        :return: a float score indicating the reward for the current turn.
        """
        # computing state feature
        with torch.no_grad():
            state = self.plm(**batch['next_state'])[0][:, 0, :]
        # objective embedding
        state = self.projector(state)
        # compute the feature representation a.k.a estimated reward
        features = self.sf(state)
        # g(s) + gamma * h(s', w(\theta)) - h(s, w(\theta))
        return features

    def manipulate_gradient_update(self, is_preference_block=True, flag=True):
        """
        method that manipulate the gradient update of each block in the model
        :param is_preference_block: True if we wish to manipulate the gradient of the preference block
        :param flag: the status of the gradient update
        :return: None
        """
        # during lr training we freeze parameters of the plm and the objective embedding layer
        # first, we freeze all parameters including the backbone plm
        # we also freeze the parameters of the projecttor
        for name, parameters in self.named_parameters():
            parameters.requires_grad_(False)

        # manipulating gradient of the preference estimation block
        # these blocks include the gs, hs, objective_embedding_layer, preference_params
        if is_preference_block:
            self.Q_head.projector.requires_grad_(flag)
            self.Q_head.objective_embedding.requires_grad_(flag)
        # manipulating the gradient update of the policy part
        # freeze or unfreeze parameters of the objective embedding, plm and out layer
        # the policy blocks involve with actor and critic models
        else:
            # self.Q_head.projector.requires_grad_(flag)
            # self.Q_head.objective_embedding.requires_grad_(flag)
            # self.Q_head.goal_net.requires_grad_(flag)
            self.Q_head.critic.requires_grad_(flag)
            self.Q_head.actor.requires_grad_(flag)

    def compute_state_resp(self, batch, w):
        """
        method that computes state representations, only used during the actor-critic training step
        :param batch: the features of the given states
        :param w: a batch of sampled preference computed using the sample_preference method.
        :return:
        """
        # no gradient here
        with torch.no_grad():
            state = self.plm(**batch['context']).last_hidden_state[:, 0, :]
            # with torch.no_grad():
            # cls token as the state
            if 'next_state' in batch:
                next_state = self.plm(**batch['next_state']).last_hidden_state[:, 0, :]
                next_state = self.Q_head.projector(next_state)
            else:
                next_state = None

        objective_embedding = self.Q_head.objective_embedding(w)
        state = self.Q_head.projector(state)
        
        # applying the dropout layer
        return state, next_state, objective_embedding

    def compute_state_value(self, feature):
        """
        function that compute the state values
        :return:
        """
        values = self.Q_head.critic(feature)
        return values

    def compute_log_probs(self, state_resp, batch_act):
        """
        method that compute the log probs
        :param state_resp: the state representations
        :return: log probs
        """
        # producing the outputs
        logits = self.Q_head.actor(state_resp)
        prob = torch.softmax(logits, dim=-1)
        dist = Categorical(prob)
        return dist.log_prob(batch_act), dist
    
    def compute_policy(self, state_resp):
        # producing the outputs
        logits = self.Q_head.actor(state_resp)
        prob = torch.softmax(logits, dim=-1)
        dist = Categorical(prob)
        act = dist.sample()
        return act