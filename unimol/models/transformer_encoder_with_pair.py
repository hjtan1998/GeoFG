# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import TransformerEncoderLayer, LayerNorm
import sys
import copy
# /home/aizoo/miniconda3/envs/alidiff/lib/python3.8/site-packages/unicore-0.0.1-py3.8.egg/unicore/modules/transformer_encoder_layer.py



class TransformerEncoderWithPair(nn.Module):
    # 段代码实现了一个名为 TransformerEncoderWithPair 的模块，它扩展了传统的 Transformer 编码器，
    # 特别关注成对的注意力表示（pair representation）和注意力矩阵的变化。
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,                      # 控制 LayerNorm 是否在注意力层和前馈层之后执行。
        no_final_head_layer_norm: bool = False,     # 控制是否禁用最后的头部 LayerNorm
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.fg_emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.fg_final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.fg_final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        fg_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fg_padding_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        


        input_padding_mask = padding_mask
        fg_input_padding_mask = fg_padding_mask

        input_cross_attn_mask = cross_attn_mask
        
        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            # 把后面的一些列变成fill_val
            # 行不能变，变了就不能attention了
            seq_len_1, seq_len_2 = attn_mask.shape[1], attn_mask.shape[2]
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(padding_mask.size(0), -1, seq_len_1, seq_len_2)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len_1, seq_len_2)
            return attn_mask
        # 这个函数 fill_attn_mask 的作用是将 padding_mask 信息合并到 attn_mask 里，
        # 用于 Transformer 这类模型在计算 self-attention 时屏蔽掉无效（padding）位置的注意力。


        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = emb
        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))


        fg_seq_len = fg_emb.size(1)
        fg_x = self.fg_emb_layer_norm(fg_emb)
        fg_x = F.dropout(fg_x, p=self.emb_dropout, training=self.training)
        # account for padding while computing the representation
        if fg_padding_mask is not None:
            fg_x = fg_x * (1 - fg_padding_mask.unsqueeze(-1).type_as(fg_x))


        def apply_mask(attention_bias, padding_mask, fill_val=float("-inf")):
            # 通过加[0,-inf]的方法，变成带有-inf的attention_bias
            mask = torch.zeros_like(attention_bias)
            seq_len_1, seq_len_2 = mask.shape[1], mask.shape[2]
            if attention_bias is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                mask = mask.view(padding_mask.size(0), -1, seq_len_1, seq_len_2)
                mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                mask = mask.view(-1, seq_len_1, seq_len_2)
            return attention_bias + mask


        assert cross_attn_mask is not None
        fg_atom_cross_attn_mask = apply_mask(cross_attn_mask, padding_mask)

        padding_mask, fg_padding_mask = None, None




        for i in range(len(self.layers)):
            x, fg_x, fg_atom_cross_attn_mask, fg_atom_attn_probs = self.layers[i](
                x=x, 
                fg_x=fg_x, 
                padding_mask=padding_mask, 
                fg_padding_mask=fg_padding_mask, 
                fg_atom_cross_attn_bias=fg_atom_cross_attn_mask,
                return_attn=True
            )
            # print("x[0,:2,:10]: ", x[0,:2,:10])       x 没变
            # if i == 4:
            #     sys.exit()
            # print("fg_x[0,:2,:10]: ", fg_x[0,:2,:10])       # fg_x 在更新
            # if i == 4:
            #     sys.exit()
            # print("x.shape: ", x.shape)
            # print("fg_x.shape: ", fg_x.shape)
            # print("fg_atom_cross_attn_mask.shape: ", fg_atom_cross_attn_mask.shape)
            # print("fg_atom_attn_probs.shape: ", fg_atom_attn_probs.shape)
            # sys.exit()
        def norm_loss(x, eps=1e-10, tolerance=1.0):
            # 计算特定向量的规范化损失，通常用于对模型输出的正则化约束。
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        fg_x_norm = norm_loss(fg_x)
        if fg_input_padding_mask is not None:
            fg_token_mask = 1.0 - fg_input_padding_mask.float()
        else:
            fg_token_mask = torch.ones_like(fg_x_norm, device=fg_x_norm.device)
        fg_x_norm = masked_mean(fg_token_mask, fg_x_norm)





        # fg-atom delta_pair_repr
        fg_atom_delta_pair_repr = fg_atom_cross_attn_mask - input_cross_attn_mask    # 计算了输入和输出的注意力掩码差异
        fg_atom_delta_pair_repr = fill_attn_mask(fg_atom_delta_pair_repr, input_padding_mask, 0)


        fg_atom_cross_attn_mask = (fg_atom_cross_attn_mask.view(bsz, -1, fg_seq_len, seq_len).permute(0, 2, 3, 1).contiguous())
        fg_atom_delta_pair_repr = (
            fg_atom_delta_pair_repr.view(bsz, -1, fg_seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        fg_atom_pair_mask = fg_token_mask[..., None] * token_mask[..., None, :]
        # torch.set_printoptions(threshold=float('inf'))
        # print("fg_token_mask: ", fg_token_mask)   # 24个，前13个有效
        # print("token_mask: ", token_mask)         # 40个，前29个有效
        # print("fg_atom_pair_mask: ", fg_atom_pair_mask)           # 维度: bts*24*40   前13行非0    前29列非0 
        # sys.exit()
        fg_atom_delta_pair_repr_norm = norm_loss(fg_atom_delta_pair_repr)
        fg_atom_delta_pair_repr_norm = masked_mean(
            fg_atom_pair_mask, fg_atom_delta_pair_repr_norm, dim=(-1, -2)
        )


        if self.fg_final_layer_norm is not None:
            fg_x = self.fg_final_layer_norm(fg_x)


        res = {
            'x': x,
            'x_norm': x_norm,

            'fg_x': fg_x,
            'fg_x_norm': fg_x_norm,

            'fg_atom_cross_attn_mask': fg_atom_cross_attn_mask,
            'fg_atom_delta_pair_repr_norm': fg_atom_delta_pair_repr_norm,
            'fg_atom_attn_probs': fg_atom_attn_probs,
        }




        # print('x.shape: ', x.shape)
        # print('attn_mask.shape: ', attn_mask.shape)
        # print('delta_pair_repr.shape: ', delta_pair_repr.shape)
        # print('x_norm: ', x_norm)
        # print('delta_pair_repr_norm: ', delta_pair_repr_norm)

        # print('fg_x.shape: ', fg_x.shape)
        # print('fg_attn_mask.shape: ', fg_attn_mask.shape)
        # print('fg_delta_pair_repr.shape: ', fg_delta_pair_repr.shape)
        # print('fg_x_norm: ', fg_x_norm)
        # print('fg_delta_pair_repr_norm: ', fg_delta_pair_repr_norm)

        # print('fg_atom_cross_attn_mask.shape: ', fg_atom_cross_attn_mask.shape)
        # print('fg_atom_delta_pair_repr.shape: ', fg_atom_delta_pair_repr.shape)
        # print('fg_atom_delta_pair_repr_norm: ', fg_atom_delta_pair_repr_norm)

        # print('fg_atom_cross_attn_mask: ', fg_atom_cross_attn_mask)
        # print('fg_atom_delta_pair_repr: ', fg_atom_delta_pair_repr)
        # sys.exit()


        return res

        # x: 编码后的输出。
        # attn_mask: 最终的注意力矩阵。
        # delta_pair_repr: 注意力矩阵的差异。
        # x_norm: 输出的规范化值。
        # delta_pair_repr_norm: 注意力差异的规范化值。

        # x
        # attn_mask
        # delta_pair_repr
        # x_norm
        # delta_pair_repr_norm

        # fg_x
        # fg_attn_mask
        # fg_delta_pair_repr
        # fg_x_norm
        # fg_delta_pair_repr_norm


        # fg_atom_cross_attn_mask
        # fg_atom_delta_pair_repr
        # fg_atom_delta_pair_repr_norm









