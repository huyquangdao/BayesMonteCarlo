import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel

from base.model import Model

from loguru import logger


class MinDistPADPPModel(Model):

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

        # freeze the paremters for the pretrained language model
        if self.model_config.freeze_plm:
            logger.info("Freezing the parameters of the pretrained language modek")
            for name, params in self.plm.named_parameters():
                params.requires_grad_(False)

        self.drop_out = nn.Dropout(p=self.model_config.dropout)

        # objective embedding
        # this layer convert a preference vector to a embedding vector for the model
        self.objective_embedding = nn.Linear(self.model_config.n_objectives,
                                             self.model_config.objective_embedding_size)

        # for negotiation, esc, we only have strategies i.goals
        n_classes = self.model_config.n_goals

        # if we predict both goal, topic at a time
        # i.e the recommendation scenario.
        if self.model_config.combined_action:
            # other parameters.
            # for recommendation, for other scenarios, new code might be needed.
            n_classes = self.model_config.n_goals * self.model_config.n_topics

        # feature function
        # which will be learned
        # Phi (s) -> R^{d}
        # self.criterion = torch.nn.CrossEntropyLoss()
        # a projector that projects the state representation to a much much lower dimension space
        self.projector = nn.Sequential(
            nn.Linear(
                self.model_config.lm_size,
                self.model_config.mlp_hidden_size * 2
            ),
            nn.ReLU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(self.model_config.mlp_hidden_size * 2,
                      self.model_config.mlp_hidden_size)
        )

        # actor model
        # we use the state embedding, context embedding and objective embedding
        # Actor(a|projector(s),w)
        # self.actor = nn.Linear(2 * self.model_config.objective_embedding_size, n_classes)
        self.actor = nn.Sequential(
            nn.Linear(
                # phi(s), w -> n_objectives
                self.model_config.mlp_hidden_size + self.model_config.objective_embedding_size,
                self.model_config.mlp_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.model_config.mlp_hidden_size * 2, self.model_config.mlp_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(
                # phi(s), w -> n_objectives
                self.model_config.mlp_hidden_size * 2,
                n_classes * self.model_config.n_objectives),
        )


    def forward(self, batch, is_pretraining=True, var=1e-2):
        """
        The forward function for class MODPL for recommendation
        :param batch:
        :param is_pretraining:
        :return:
        """
        # predicting the policy
        # using the state and objective embedding
        # objective embedding
        assert is_pretraining is True
        if is_pretraining:
            # cls token as the state
            state = self.plm(**batch['context'])[0][:, 0, :]
            bs = state.shape[0]
            
            # preference embedding
            objective_embedding = self.objective_embedding(batch['w'])
            feature = state
            feature = self.projector(feature)
            # 2 x objective embedding size

            feature = torch.cat([feature, objective_embedding], dim=-1)
            # computing the logits using the current state
            # using the actor
            # concatenating the state, context vector and objective embedding
            # a ~ \pi(.|s,w)

            logits = self.actor(feature)
            logits = logits.view(bs, -1, self.model_config.n_objectives)
            logits = torch.bmm(logits, batch['w'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)
            return logits

        else:
            raise Exception("Forward function only used for pretraining")

    def compute_features(self, batch):
        """
        function that estimate the reward for a particular state and preference
        :return: a float score indicating the reward for the current turn.
        """
        # computing state feature
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
            self.sf.requires_grad_(flag)
        # manipulating the gradient update of the policy part
        # freeze or unfreeze parameters of the objective embedding, plm and out layer
        # the policy blocks involve with actor and critic models
        else:
            self.objective_embedding.requires_grad_(flag)
            self.projector.requires_grad_(flag)
            self.critic.requires_grad_(flag)
            self.actor.requires_grad_(flag)

    def compute_state_resp(self, batch, w):
        """
        method that computes state representations, only used during the actor-critic training step
        :param batch: the features of the given states
        :param w: a batch of sampled preference computed using the sample_preference method.
        :return:
        """
        # no further gradient update on the objective embedding or the backbone plm
        with torch.no_grad():
            state = self.plm(**batch['context'])[0][:, 0, :]
            # cls token as the state
            if 'next_state' in batch:
                next_state = self.plm(**batch['next_state'])[0][:, 0, :]
                next_state = self.projector(next_state)
            else:
                next_state = None

        # only update the objective embedding and the projector
        objective_embedding = self.objective_embedding(w)
        state = self.projector(state)

        # applying the dropout layer
        return state, next_state, objective_embedding

    def compute_state_value(self, state, objective_embedding):
        """
        function that compute the state values
        :return:
        """
        state = self.projector(state)
        # compute the output of feature representation
        feature_resp = self.sf(state)
        # g(s) + gamma * h(s', w(\theta)) - h(s, w(\theta))
        inp = torch.cat([feature_resp, objective_embedding], dim=-1)
        # compute the values for critic
        values = self.critic(inp)
        return feature_resp, values

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
