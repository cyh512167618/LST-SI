import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class MIND(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(MIND, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):

        mask = self.get_attention_mask(item_seq)
        item_his_eb = self.item_embedding(item_seq)
        a = mask.view(-1, 50, 1)
        item_his_eb = item_his_eb * mask.view(256, 50, -1)

        item_emb = self.item_embedding(item_seq)
        capsule_network = CapsuleNetwork(self.hidden_size, 50, bilinear_type=0,
                                         num_interest=4,
                                         hard_readout=True, relu_layer=False)
        user_eb, readout = capsule_network(item_his_eb, item_emb, mask)
        return user_eb


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class CapsuleNetwork(nn.Module):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.device = device
        self.linear = nn.Linear(64,self.dim,bias=False).to(self.device)

    def forward(self, item_his_emb, item_eb, mask):
        if self.bilinear_type == 0:
            item_emb_hat = self.linear(item_his_emb.to(self.device))
            item_emb_hat = item_emb_hat.repeat(1, 1, self.num_interest)
        elif self.bilinear_type == 1:
            item_emb_hat = nn.Linear(item_his_emb.size(2), self.dim * self.num_interest, bias=False)(item_his_emb.to(self.device))
        else:
            w = nn.Parameter(torch.randn(1, self.seq_len, self.num_interest * self.dim, self.dim))
            u = item_his_emb.unsqueeze(2)
            item_emb_hat = torch.sum(w[:, :self.seq_len, :, :] * u, dim=3)

        item_emb_hat = item_emb_hat.view(-1, self.seq_len, self.num_interest, self.dim)
        item_emb_hat = item_emb_hat.transpose(1, 2).contiguous()
        item_emb_hat = item_emb_hat.view(-1, self.num_interest, self.seq_len, self.dim)

        if self.stop_grad:
            item_emb_hat_iter = item_emb_hat.detach()
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = nn.Parameter(torch.zeros(item_his_emb.size(0), self.num_interest, self.seq_len))
        else:
            capsule_weight = nn.Parameter(torch.randn(item_his_emb.size(0), self.num_interest, self.seq_len))

        for i in range(3):
            atten_mask = mask.unsqueeze(1).repeat(1, self.num_interest, 1)
            paddings = torch.zeros_like(atten_mask)

            capsule_softmax_weight = nn.functional.softmax(capsule_weight, dim=1).to(self.device)
            mask = torch.eq(atten_mask, 0.0)
            capsule_softmax_weight = atten_mask * paddings + (1 - mask) * capsule_softmax_weight
            capsule_softmax_weight = capsule_softmax_weight.unsqueeze(2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = torch.sum(interest_capsule.pow(2), dim=-1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_emb_hat_iter, interest_capsule.transpose(2, 3))
                delta_weight = delta_weight.view(-1, self.num_interest, self.seq_len)
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = torch.sum(interest_capsule.pow(2), dim=-1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = interest_capsule.view(-1, self.num_interest, self.dim)

        if self.relu_layer:
            interest_capsule = nn.Linear(interest_capsule.size(2), self.dim)(interest_capsule)

        atten = torch.matmul(interest_capsule, item_eb.view(-1, self.dim, 1))
        atten = nn.functional.softmax(atten.view(-1, self.num_interest).pow(1), dim=1)

        if self.hard_readout:
            readout = interest_capsule.view(-1, self.dim)[
                torch.arange(item_his_emb.size(0)) * self.num_interest + atten.argmax(dim=1)]
        else:
            readout = torch.matmul(atten.view(item_his_emb.size(0), 1, self.num_interest), interest_capsule)
            readout = readout.view(item_his_emb.size(0), self.dim)

        return interest_capsule, readout


