# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from unicore import utils
from torch import nn
from . import LayerNorm, SelfMultiheadAttention, CrossMultiheadAttention
import sys

class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)


        # 跨注意力机制  把原子信息转移到官能团
        self.fg_atom_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        self.fg_atom_attn_layer_norm = LayerNorm(self.embed_dim) # layer norm associated with the self attention layer


        # 残差层  官能团
        self.fg_fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fg_fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.fg_final_layer_norm = LayerNorm(self.embed_dim)
        self.fg_post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        fg_x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fg_padding_mask: Optional[torch.Tensor] = None,
        fg_atom_cross_attn_bias: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """


        # 跨注意力机制  把原子信息转移到官能团
        fg_residual = fg_x
        if not self.fg_post_ln:
            fg_x = self.fg_atom_attn_layer_norm(fg_x)
        # new added
        fg_x = self.fg_atom_attn(
            query=fg_x,
            key=x,
            value=x,
            key_padding_mask=None,
            attn_bias=fg_atom_cross_attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            fg_x, fg_atom_attn_weights, fg_atom_attn_probs = fg_x
        fg_x = F.dropout(fg_x, p=self.dropout, training=self.training)
        fg_x = fg_residual + fg_x
        if self.fg_post_ln:
            fg_x = self.fg_atom_attn_layer_norm(fg_x)

        # print("fg_x: ", fg_x)
        # sys.exit()


        # 残差层  官能团
        fg_residual = fg_x
        if not self.fg_post_ln:
            fg_x = self.fg_final_layer_norm(fg_x)
        fg_x = self.fg_fc1(fg_x)
        fg_x = self.activation_fn(fg_x)
        fg_x = F.dropout(fg_x, p=self.activation_dropout, training=self.training)
        fg_x = self.fg_fc2(fg_x)
        fg_x = F.dropout(fg_x, p=self.dropout, training=self.training)
        fg_x = fg_residual + fg_x
        if self.fg_post_ln:
            fg_x = self.fg_final_layer_norm(fg_x) 


        if not return_attn:
            return x, fg_x
        else:
            return x, fg_x, fg_atom_attn_weights, fg_atom_attn_probs


