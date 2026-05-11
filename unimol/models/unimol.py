# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List
from .unimol_ini import UniMolModelBase
import sys
logger = logging.getLogger(__name__)
import time
import os

def get_mol_rep_from_atoms(encoder_rep: torch.Tensor, atom_tokens: torch.Tensor) -> torch.Tensor:
    """
    从 encoder_rep 和 atom_tokens 中提取原子embedding并做平均，得到分子级表示。

    Args:
        encoder_rep: torch.Tensor, shape [B, L, D]
        atom_tokens: torch.Tensor, shape [B, L]
            每一行形式为 [1, atom_1, atom_2, ..., atom_n, 2, 0, 0, ...]

    Returns:
        mol_rep: torch.Tensor, shape [B, D]
            每个分子的原子embedding平均值（不包含CLS、SEP、PAD）
    """
    # mask: 仅保留原子token
    atom_mask = (atom_tokens != 0) & (atom_tokens != 1) & (atom_tokens != 2)  # [B, L]

    # 避免除0：每个分子原子数
    atom_counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

    # 扩展mask到embedding维度，用于加权求和
    atom_mask_expanded = atom_mask.unsqueeze(-1).float()  # [B, L, 1]

    # 仅保留原子embedding并求平均
    mol_rep = (encoder_rep * atom_mask_expanded).sum(dim=1) / atom_counts  # [B, D]

    return mol_rep



@register_model("unimol")
class UniMolModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-fg-token-loss",
            type=float,
            metavar="D",
            help="mask fg loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-relation-loss",             # masked_relation_loss
            type=float,
            metavar="D",
            help="masked relation loss ratio",
        )
        parser.add_argument(
            "--masked-fg-dist-loss",
            type=float,
            metavar="D",
            help="masked fg distance loss ratio",
        )
        parser.add_argument(
            "--masked-fg-atom-dist-loss",
            type=float,
            metavar="D",
            help="masked fg atom distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--masked-fg-coord-loss",
            type=float,
            metavar="D",
            help="masked fg coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--x-fg-norm-loss",
            type=float,
            metavar="D",
            help="x fg norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--delta-fg-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder fg pair repr norm loss ratio",
        )
        parser.add_argument(
            "--delta-fg-atom-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder fg atom pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )




    def __init__(self, args, atom_dictionary, fg_dictionary):
        super().__init__()
        base_architecture(args)

        self.mol_model = UniMolModelBase(args, atom_dictionary)


        self.args = args
        self.padding_idx = atom_dictionary.pad()
        self.fg_embed_tokens = nn.Embedding(
            len(fg_dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.fg_encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        K = 128

        self.fg_atom_gbf_proj = NonLinearHead(K, args.encoder_attention_heads, args.activation_fn)
        self.fg_atom_gbf = GaussianLayer(K, len(fg_dictionary)*len(atom_dictionary))

        
        self.classification_heads = nn.ModuleDict()
        
        # mol_param_ids = {id(p) for p in self.mol_model.parameters()}
        # for param in self.parameters():
        #     if id(param) not in mol_param_ids:
        #         param.requires_grad = False
        self.apply(init_bert_params)
 

        # print("args: ", args)
        # sys.exit()
        # 冻结模型的参数
        mol_model_data = torch.load("./data/mol_pre_no_h_220816.pt")
        self.mol_model.load_state_dict(mol_model_data['model'], strict=False)
        if args.task=='mol_finetune':    # 微调的时候不冻结
            for param in self.mol_model.parameters():
                param.requires_grad = True
        else:
            # self.mol_model.eval()
            for param in self.mol_model.parameters():
                param.requires_grad = False


        # if args.masked_relation_loss > 0:
        #     self.relation_head = MaskRelationClsHead(
        #         args.encoder_attention_heads, 2, args.activation_fn
        #     )

        if args.masked_fg_atom_dist_loss > 0:
            self.fg_atom_dist_head = CrossDistanceHead(self.args.encoder_attention_heads, self.args.activation_fn) 


        # print("self.mol_model.encoder.layers[0].self_attn.in_proj.weight: ", self.mol_model.encoder.layers[0].self_attn.in_proj.weight)
        # print("self.mol_model.encoder.layers[2].self_attn.in_proj.weight: ", self.mol_model.encoder.layers[2].self_attn.in_proj.weight)






    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.atom_dictionary, task.fg_dictionary)

    def forward(
        self,
        smi_name,

        atom_tokens,
        atom_coord,
        atom_distance,
        atom_edge_type,

        fg_tokens,
        fg_coord,
        fg_distance,
        fg_edge_type,

        fg_atom_distance,
        fg_atom_edge_type,
        fg_atom_relation,


        encoder_masked_fg_atom_relation=None,
        encoder_atom_masked_tokens=None,
        encoder_fg_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        

        
        # print("fg_atom_relation.shape: ", fg_atom_relation.shape)
        # print("encoder_masked_fg_atom_relation.shape: ", encoder_masked_fg_atom_relation.shape)
        #         
        # print(self.mol_model.training)
        # print("self.mol_model.encoder.layers[0].self_attn.in_proj.weight: ", self.mol_model.encoder.layers[0].self_attn.in_proj.weight)
        # print("self.mol_model.encoder.layers[2].self_attn.in_proj.weight: ", self.mol_model.encoder.layers[2].self_attn.in_proj.weight)
        # sys.exit()

        # print("atom_tokens.shape: ", atom_tokens.shape)
        # print("atom_coord.shape: ", atom_coord.shape)
        # print("atom_distance.shape: ", atom_distance.shape)
        # print("atom_edge_type.shape: ", atom_edge_type.shape)

        # print("fg_tokens.shape: ", fg_tokens.shape)
        # print("fg_coord.shape: ", fg_coord.shape)
        # print("fg_distance.shape: ", fg_distance.shape)
        # print("fg_edge_type.shape: ", fg_edge_type.shape)

        # print("fg_atom_distance.shape: ", fg_atom_distance.shape)
        # print("fg_atom_edge_type.shape: ", fg_atom_edge_type.shape)
        # sys.exit()



        atom_padding_mask = atom_tokens.eq(self.mol_model.padding_idx)
        if not atom_padding_mask.any():
            atom_padding_mask = None
        atom_x = self.mol_model.embed_tokens(atom_tokens)



        def get_dist_features(dist, et):
            # print("dist shape in get_dist_features: ", dist.shape)     # torch.Size([4, 120, 120])
            # print("et shape in get_dist_features: ", et.shape)         # torch.Size([4, 120, 120])
            n_node = dist.size(-1)
            gbf_feature = self.mol_model.gbf(dist, et)
            # print("gbf_feature shape in get_dist_features: ", gbf_feature.shape)   # torch.Size([4, 120, 120, 128])
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            # print("gbf_result shape in get_dist_features: ", gbf_result.shape)   # torch.Size([4, 120, 120, 64])
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        atom_graph_attn_bias = get_dist_features(atom_distance, atom_edge_type)

        (
            atom_encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.mol_model.encoder(atom_x, padding_mask=atom_padding_mask, attn_mask=atom_graph_attn_bias)



        if classification_head_name is not None:
            features_only = True


        fg_padding_mask = fg_tokens.eq(self.padding_idx)
        if not fg_padding_mask.any():
            fg_padding_mask = None
        fg_x = self.fg_embed_tokens(fg_tokens)


        def get_fg_atom_dist_features(dist, et):
            n_fg, n_atom = dist.shape[1], dist.shape[2]
            gbf_feature = self.fg_atom_gbf(dist, et)
            # print("gbf_feature.shape: ", gbf_feature.shape)               # torch.Size([256, 32, 56, 128])
            gbf_result = self.fg_atom_gbf_proj(gbf_feature)
            # print("gbf_result.shape: ", gbf_result.shape)                 # torch.Size([256, 32, 56, 64])
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_fg, n_atom)
            # print("graph_attn_bias.shape: ", graph_attn_bias.shape)       # torch.Size([16384, 32, 56])
            return graph_attn_bias
        fg_atom_graph_attn_bias = get_fg_atom_dist_features(fg_atom_distance, fg_atom_edge_type)
        # sys.exit()



        res = self.encoder(
            atom_encoder_rep, 
            fg_x, 
            padding_mask=atom_padding_mask, 
            fg_padding_mask=fg_padding_mask,
            cross_attn_mask=fg_atom_graph_attn_bias,
        )
        # 原子特征进去出来没变


        atom_encoder_rep = res["x"]
        fg_encoder_rep = res["fg_x"]
        fg_x_norm = res["fg_x_norm"]
        fg_atom_pair_rep = res["fg_atom_cross_attn_mask"]
        fg_atom_delta_pair_repr_norm = res["fg_atom_delta_pair_repr_norm"]
        fg_atom_attn_probs = res["fg_atom_attn_probs"]



        # ===============================
        # 保存 fg_tokens（每次 forward）
        # ===============================
        # ts = time.time_ns()   # 纳秒级时间戳
        # save_path = os.path.join('/home/aizoo/data/workspace/tanhaojiang/AI4Sci/Uni-Mol-main/unimol/save_finetune/pretrained_mol_20260111_132001/fg_emb/bace', f"fg_tokens_{ts}.pt")
        # torch.save(
        #     {
        #         "smi_name": smi_name,
        #         "fg_tokens": fg_tokens.detach().cpu(),
        #         "fg_encoder_rep": fg_encoder_rep.detach().cpu(),
        #     },
        #     save_path
        # )




        atom_logits = atom_encoder_rep
        fg_logits = fg_encoder_rep
        fg_atom_pair_rep[fg_atom_pair_rep == float("-inf")] = 0
        # print("fg_atom_pair_rep.shape: ", fg_atom_pair_rep.shape)

        selected_rep = fg_atom_pair_rep[encoder_masked_fg_atom_relation]
        # print("selected_rep.shape: ", selected_rep.shape)

        # print("self.args.masked_relation_loss: ",self.args.masked_relation_loss)
        # encoder_relation = None
        # if self.args.masked_relation_loss > 0:
        #     encoder_relation = self.relation_head(selected_rep)

        # print("encoder_relation.shape: ", encoder_relation.shape)
        # sys.exit()

        fg_atom_distance_pre = None
        if self.args.masked_fg_atom_dist_loss > 0:
            fg_atom_distance_pre = self.fg_atom_dist_head(fg_atom_pair_rep) 
            fg_atom_distance_pre = fg_atom_distance_pre[encoder_masked_fg_atom_relation]     

        # print("classification_head_name: ", classification_head_name)    # pertrain是None，finetune是bace(数据集)
        if classification_head_name is not None:
            fg_rep = get_mol_rep_from_atoms(fg_encoder_rep, fg_tokens)
            fg_encoder_rep[:, 0, :] = fg_rep
            atom_logits = self.classification_heads[classification_head_name](atom_encoder_rep, fg_encoder_rep)
            fg_logits = None
        
        
        # print("self.args.mode: ", self.args.mode)   # 无论pertrain还是finetune 都是self.args.mode:  train       
        if self.args.mode == 'infer':
            return atom_encoder_rep, encoder_pair_rep
        else:
            # print("logits.shape: ", logits.shape)
            return (
                atom_logits,
                fg_logits,
                fg_x_norm,
                fg_atom_delta_pair_repr_norm,
                fg_atom_distance_pre,
            )



    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class MaskRelationClsHead(nn.Module):
    def __init__(self, embed_dim, num_classes, activation_fn="gelu", dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, features, masked_tokens=None):
        """
        features: [N, D] or [B, ..., D]
        masked_tokens: optional bool mask, same leading shape as features
        """
        if masked_tokens is not None:
            features = features[masked_tokens]   # [N_masked, D]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)              # [N_masked, num_classes]
        return logits


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim*2, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, atom_features, fg_features, **kwargs):
        atom_x = atom_features[:, 0, :]  # take <s> token (equiv. to [CLS])
        fg_x = fg_features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = torch.cat([atom_x, fg_x], dim=-1)
        # x = atom_x
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x



class CrossDistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len_1, seq_len_2, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len_1, seq_len_2)
        return x



@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        # print("x.shape: ", x.shape)          # torch.Size([256, 32, 56, 1])
        x = x.expand(-1, -1, -1, self.K)
        # print("x.shape: ", x.shape)          # torch.Size([256, 32, 56, 128])
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class RelationalGaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

        self.belong_embed = nn.Embedding(5, K)  # 0: padding, 1: start end token 2:非所属  3:所属    4:mask
        nn.init.xavier_uniform_(self.belong_embed.weight)

        self.belong_gate = nn.Sequential(
                nn.Linear(K+1, 1),
                nn.Sigmoid()
            )

    def forward(self, x, edge_type, fg_atom_relation):

        # print("x.shape: ", x.shape)                    # torch.Size([256, 32, 56])
        # print("edge_type.shape: ", edge_type.shape)      # torch.Size([256, 32, 56])      
        # print("fg_atom_relation.shape in RelationalGaussianLayer: ", fg_atom_relation.shape)       # torch.Size([256, 32, 56])
        
        belong_embed = self.belong_embed(fg_atom_relation.long())
        # print("belong_embed.shape: ", belong_embed.shape)                    # torch.Size([256, 32, 56, 128])
        belong_context = torch.cat([belong_embed, x.unsqueeze(-1).type_as(belong_embed)], dim=-1)
        # print("belong_context.shape: ", belong_context.shape) # torch.Size([256, 32, 56, 129])
        belong_gate = self.belong_gate(belong_context)
        # torch.set_printoptions(threshold=float('inf'))
        # print("belong_gate.shape: ", belong_gate.shape)  # torch.Size([256, 32, 56, 1])
        # print("fg_atom_relation[0]: ", fg_atom_relation[0])  # torch.Size([256, 32, 56, 1])
        # print("belong_gate[0]: ", belong_gate[0])  # torch.Size([256, 32, 56, 1])
        # sys.exit()
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        # print("x.shape: ", x.shape)          # torch.Size([256, 32, 56, 1])
        x = x * belong_gate
        x = x.expand(-1, -1, -1, self.K)
        # print("x.shape: ", x.shape)          # torch.Size([256, 32, 56, 128])
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)




@register_model_architecture("unimol", "unimol")
def base_architecture(args):
    args.fg_encoder_layers = getattr(args, "fg_encoder_layers", 15)
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)               # 512
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)      # 2048
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.masked_relation_loss = getattr(args, "masked_relation_loss", -1.0)
    args.masked_fg_token_loss = getattr(args, "masked_fg_token_loss", -1.0)
    args.masked_fg_coord_loss = getattr(args, "masked_fg_coord_loss", -1.0)
    args.masked_fg_dist_loss = getattr(args, "masked_fg_dist_loss", -1.0)
    args.masked_fg_atom_dist_loss = getattr(args, "masked_fg_atom_dist_loss", -1.0)

    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.x_fg_norm_loss = getattr(args, "x_fg_norm_loss", -1.0)
    args.delta_fg_pair_repr_norm_loss = getattr(args, "delta_fg_pair_repr_norm_loss", -1.0)
    args.delta_fg_atom_pair_repr_norm_loss = getattr(args, "delta_fg_atom_pair_repr_norm_loss", -1.0)




@register_model_architecture("unimol", "unimol_base")
def unimol_base_architecture(args):
    base_architecture(args)













