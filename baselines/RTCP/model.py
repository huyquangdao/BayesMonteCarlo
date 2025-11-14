import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from base.model import Model
from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.contiguous().view(-1, size[-1])).view(size)


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        """
        class Transformer Feed Forward Neural Net
        :param dim: the input dimension
        :param dim_hidden: the hidden dimension
        :param relu_dropout: dropout rate
        """
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            dropout=0.0,
    ):
        """
        class Cross-Attention in RTCP model
        :param n_heads: the number of heads
        :param embedding_size: the embedding dim
        :param ffn_size: the ffn dim
        :param dropout: dropout rate
        """
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size

        self.attention = MultiheadAttention(
            self.dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, q, k, v, mask):
        output, _ = self.attention(q, k, v, key_padding_mask=mask)
        return output


class CrossEncoderLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        """
        class Cross Encoder layer
        :param n_heads: the number of heads in the attention block
        :param embedding_size: the embedding dim
        :param ffn_size: the feedforward neural net dim
        :param attention_dropout: attention drop rate
        :param relu_dropout: rele drop rate
        :param dropout: dropout rate
        """
        super().__init__()
        self.output_attention = CrossAttentionLayer(
            n_heads,
            embedding_size,
            ffn_size=ffn_size,
            dropout=attention_dropout,
        )

        self.context_attention = CrossAttentionLayer(
            n_heads,
            embedding_size,
            ffn_size=ffn_size,
            dropout=attention_dropout,
        )

        self.knowledge_attention = CrossAttentionLayer(
            n_heads,
            embedding_size,
            ffn_size=ffn_size,
            dropout=attention_dropout,
        )

        self.path_attention = CrossAttentionLayer(
            n_heads,
            embedding_size,
            ffn_size=ffn_size,
            dropout=attention_dropout,
        )

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

        self.ffn_o = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn_c = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn_p = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.ffn_k = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)

    def forward(self, o, c, p, k, c_mask, p_mask, k_mask):
        ### o output from the previous layer
        ### k knowledge latents
        ### c context latents
        ### p profile latents

        ### multi-head attention for context
        ### attent to it self.
        o = self.output_attention(o, o, o, (1 - p_mask).bool())

        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_o(o))
        o = _normalize(o, self.norm2)
        # o *= c_mask.unsqueeze(-1).type_as(o)

        ### cross multihead attention for path
        o = self.path_attention(o, p, p, (1 - p_mask).bool())
        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_p(o))
        o = _normalize(o, self.norm2)

        ### cross_multi_head_attention for context
        o = self.context_attention(o, c, c, (1 - c_mask).bool())
        o = _normalize(o, self.norm1)
        o = o + self.dropout(self.ffn_c(o))
        o = _normalize(o, self.norm2)
        # o *= p_mask.unsqueeze(-1).type_as(o)

        # since knowledge is not available therefore we ignore the knowledge attention part.
        # cross_multi_head_attention for output and knowledge
        # o = self.knowledge_attention(o, k, k, (1 - k_mask).bool())
        # o = _normalize(o, self.norm1)
        # o = o + self.dropout(self.ffn_k(o))
        # o = _normalize(o, self.norm2)
        # o *= k_mask.unsqueeze(-1).type_as(o)

        return o


class CrossEncoder(nn.Module):
    def __init__(
            self,
            n_layers,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        """
        class Cross Encoder
        :param n_layers: the number of attention layers
        :param n_heads: the number of heads in the attention blocks
        :param embedding_size: the embedding size
        :param ffn_size: the ffn size
        :param attention_dropout: attention dropout rate
        :param relu_dropout: relu dropout rate
        :param dropout: dropout rate
        """
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(CrossEncoderLayer(
                n_heads,
                embedding_size,
                ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, inputs):

        ### inputs = [context_latents, knowledge_latents, profile_latents, context_mask, knowledge_mask, profile_mask]
        ### initialize the output with the path latent
        x = inputs['path_latent']

        # forward pass
        for layer in self.layers:
            # forward the input through each layer
            x = layer(
                x,
                inputs['context_latent'],
                inputs['path_latent'],
                inputs['knowledge_latent'],
                inputs['context_mask'],
                inputs['path_mask'],
                inputs['knowledge_mask']
            )

        return x


class RTCPModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for the RTCPModel
        :param model_config: the model configuration
        :param kwargs: other keywords args
        """
        super().__init__(model_config, **kwargs)
        # create the tokenizer and the backbone pretrained language model
        # for rtcop, we need to initialize a context encoder and path encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)
        self.context_encoder = AutoModel.from_pretrained(self.model_config.plm, cache_dir=self.model_config.cached_dir)
        self.path_encoder = AutoModel.from_pretrained(self.model_config.plm, cache_dir=self.model_config.cached_dir)

        # prepend special tokens to the vocabulary and resize the embedding matrix of the context and  path
        self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
        self.context_encoder.resize_token_embeddings(len(self.tokenizer))
        self.path_encoder.resize_token_embeddings(len(self.tokenizer))

        self.fc_hidden_size = self.model_config.fc_hidden_size
        # create the cross encoder model
        self.cross_encoder_model = CrossEncoder(
            self.model_config.n_layers,
            self.model_config.n_heads,
            self.model_config.lm_size,
            self.model_config.ffn_size,
            attention_dropout=self.model_config.attention_dropout,
            relu_dropout=self.model_config.relu_dropout,
            dropout=self.model_config.dropout,
        )
        # rtcp for the recommendation scenario
        if self.model_config.scenario_name == RECOMMENDATION:
            # the hidden and out layers for goal prediction
            self.goal_fc = nn.Linear(self.model_config.lm_size, self.model_config.fc_hidden_size)
            self.goal_out_layer = nn.Linear(self.model_config.fc_hidden_size, self.model_config.n_goals)
            # the hidden and out layers for topic prediction
            self.topic_fc = nn.Linear(self.model_config.lm_size, self.model_config.fc_hidden_size)
            self.topic_out_layer = nn.Linear(self.model_config.fc_hidden_size, self.model_config.n_topics)
        # negotiation and emotional support conversation
        elif self.model_config.scenario_name in [NEGOTIATION, EMOTIONAL_SUPPORT]:
            # the hidden and out layers for goal prediction
            self.goal_fc = nn.Linear(self.model_config.lm_size, self.model_config.fc_hidden_size)
            self.goal_out_layer = nn.Linear(self.model_config.fc_hidden_size, self.model_config.n_goals)

    def forward(self, batch):
        """
        the forward function for the RTCP policy model
        :param batch: the input data
        :return:
        """
        # the latent embedding of the dialogue context
        context_latents = self.context_encoder(**batch['context'])[0]

        # currently, knowledge part is not available for interactive evaluation
        # knowledge_latents = self.knowledge_encoder(input_ids=inputs['knowledge'][0],
        #                                            token_type_ids=inputs['knowledge'][1],
        #                                            position_ids=inputs['knowledge'][2],
        #                                            attention_mask=inputs['knowledge'][3],
        #                                            )[0]

        # the latent embedding of the previous planned path
        path_latents = self.path_encoder(**batch['path'])[0]
        out_dict = {
            "context_latent": context_latents,
            "knowledge_latent": context_latents,  # still pass but did not use to prevent a buch of code modifications
            "path_latent": path_latents,
            "context_mask": batch['context']['attention_mask'],
            "knowledge_mask": batch['context']['attention_mask'],
            # still pass but did not use to prevent a bunch of code modidications
            "path_mask": batch['context']['attention_mask']
        }

        # compute the output using the cross encoder model
        output = self.cross_encoder_model(out_dict)
        cls_tokens = output[:, 0, :]
        goal_logits = self.goal_out_layer(torch.relu(self.goal_fc(cls_tokens)))

        # goal prediction loss and accuracy
        ce_loss = CrossEntropyLoss()
        goal_loss = ce_loss(goal_logits, batch['labels_goal'])

        # for recommendation scneario, we need to predict the topic also
        if self.model_config.scenario_name == RECOMMENDATION:
            # topic prediction and loss
            topic_logits = self.topic_out_layer(torch.relu(self.topic_fc(cls_tokens)))
            topic_loss = ce_loss(topic_logits, batch['labels_topic'])
            return goal_logits, topic_logits, goal_loss + topic_loss,

        return goal_logits, goal_loss
