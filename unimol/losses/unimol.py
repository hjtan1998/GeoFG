# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import sys




# Pharmacophore 分组字典
pharmacophore_fg_dict = {
    "Donor": ["SingleAtomDonor","Imidazole","Guanidine"],
    "Acceptor": ["SingleAtomAcceptor","Imidazole","Nitro2"],
    "Aromatic": ["Imidazole","Arom4", "Arom5", "Arom6", "Arom7", "Arom8","RH6_6", "RH5_5", "RH4_4", "RH3_3"],
    "PosIonizable": ["BasicGroup","PosN","Imidazole","Guanidine"],
    "NegIonizable": ["AcidicGroup"],
    "Hydrophobe": ["ThreeWayAttach","ChainTwoWayAttach"],
    "ZnBinder": ["ZnBinder1", "ZnBinder2", "ZnBinder3", "ZnBinder4", "ZnBinder5", "ZnBinder6"],
    "LumpedHydrophobe": ["Nitro2",
        "Arom4", "Arom5", "Arom6", "Arom7", "Arom8",
        "RH6_6", "RH5_5", "RH4_4", "RH3_3",
        "tButyl","iPropyl"
    ]
}


def build_fg_group_matrix(fg_dictionary, pharmacophore_fg_dict):
    group_names = list(pharmacophore_fg_dict.keys())
    num_groups = len(group_names)
    vocab_size = len(fg_dictionary)

    fg_group = torch.zeros(vocab_size, num_groups, dtype=torch.bool)

    for g_id, group_name in enumerate(group_names):
        for fg_name in pharmacophore_fg_dict[group_name]:
            if fg_name in fg_dictionary:
                fg_id = fg_dictionary.index(fg_name)
                fg_group[fg_id, g_id] = True

    return fg_group, group_names





@register_loss("unimol")
class UniMolLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.fg_dictionary_len = len(task.fg_dictionary)
        self.fg_dictionary = task.fg_dictionary

        # print("self.fg_dictionary: ", self.fg_dictionary)
        # print("self.fg_dictionary: ", self.fg_dictionary.index("Arom4"))       # 得到index
        # print("self.fg_dictionary: ", self.fg_dictionary.__getitem__(6))       # 得到index为6的官能团
        # print("self.fg_dictionary: ", self.fg_dictionary.special_index())      # [31, 3, 0, 1, 2]
        # sys.exit()

        self.padding_idx = task.atom_dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

        self.fg_group_matrix, self.group_names = build_fg_group_matrix(
            self.fg_dictionary,
            pharmacophore_fg_dict
        )

        

    def forward(self, model, sample, reduce=True):
        # print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
        # print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M")
        input_key = "net_input"

        target_key = "target"
        masked_fg_atom_relation = sample[target_key]["fg_atom_relation_target"].ne(self.padding_idx)   # ne = not equal（不等于） 
        # print("masked_fg_atom_relation: ", torch.sum(masked_fg_atom_relation[0]))
        # print("masked_fg_atom_relation: ", masked_fg_atom_relation[0])
        # sys.exit()

        sample_size = masked_fg_atom_relation.long().sum()
        masked_cnt = sample_size

        fg_atom_relation_target = sample[target_key]["fg_atom_relation_target"]

        # torch.set_printoptions(threshold=float('inf'))
        # print("masked_fg_atom_relation[1]: ", masked_fg_atom_relation[1])
        # print("sample[input_key]['fg_atom_relation'][1]: ", sample[input_key]['fg_atom_relation'][1])
        # print("fg_atom_relation_target[1]: ", fg_atom_relation_target[1])
        # print("sample[input_key]['fg_atom_distance'][1]: ", sample[input_key]['fg_atom_distance'][1])
        


        

        # 引入距离噪声
        dist = sample[input_key]['fg_atom_distance']          # (B, F, A)
        fg_atom_dist_target = dist[masked_fg_atom_relation]
        # 1. 生成噪声
        # noise = torch.empty_like(dist).uniform_(-1.0, 1.0)      # 均匀噪声
        noise = torch.empty_like(dist).uniform_(-5, 5)      # 均匀噪声
        # 2. 只在 mask=True 的位置加噪声
        dist_noisy = dist.clone()
        dist_noisy[masked_fg_atom_relation] = dist[masked_fg_atom_relation] + noise[masked_fg_atom_relation]
        # 3. 保证距离 > 0（数值安全）
        dist_noisy = torch.clamp(dist_noisy, min=1e-6)
        # 4. 写回（可选）
        sample[input_key]['fg_atom_distance'] = dist_noisy

        # print("sample[input_key]['fg_atom_distance'][1]: ", sample[input_key]['fg_atom_distance'][1])
        # print("fg_atom_dist_target: ", fg_atom_dist_target[1:10])
        # print("dist_noisy: ", dist_noisy[masked_fg_atom_relation][1:10])
        # sys.exit()

        (
            atom_logits,
            fg_logits,
            fg_x_norm,
            fg_atom_delta_pair_repr_norm,
            fg_atom_distance_pre,
        ) = model(smi_name = sample["smi_name"], **sample[input_key], encoder_masked_fg_atom_relation=masked_fg_atom_relation, encoder_atom_masked_tokens=None, encoder_fg_masked_tokens=None)





        fg_embed_tokens = model.fg_embed_tokens.weight        # torch.Size([32, 512])

        fg_tokens = sample[input_key]['fg_tokens']            # [8, 24]
        fg_atom_relation = sample[input_key]['fg_atom_relation']  # [8, 24, 56]
        # torch.set_printoptions(threshold=torch.inf)


        # # 1. fg_tokens -> one-hot
        # fg_onehot = F.one_hot(fg_tokens, num_classes=self.fg_dictionary_len).float()
        # # 2. fg_atom_relation 对齐维度
        # fg_atom_relation = fg_atom_relation.float()    # [bts, 24, 56]
        # # 3. FG -> atom 映射
        # atom_fg_type = torch.einsum(
        #     'bkt,bka->bat',
        #     fg_onehot,
        #     fg_atom_relation
        # )
        # scaling = atom_logits.size(-1) ** -0.5
        # logits = torch.matmul(atom_logits * scaling, fg_embed_tokens.t())
        # atom_fg_type = atom_fg_type[:,:,4:self.fg_dictionary_len-1]
        # logits = logits[:,:,4:self.fg_dictionary_len-1]
        # # print("atom_fg_type.shape: ", atom_fg_type.shape)
        # # print("logits.shape: ", logits.shape)

        # eps = 1e-8
        # # 1. target 概率
        # row_sum = atom_fg_type.sum(dim=-1, keepdim=True)
        # # print("row_sum: ", row_sum)
        # valid_mask = (row_sum > 0)
        # target_prob = atom_fg_type / (row_sum + eps)
        # # 2. pred 概率（log）
        # pred_log_prob = torch.log_softmax(logits, dim=-1)
        # # 3. KL
        # target_log_prob = torch.log(target_prob + eps)
        # kl_per_atom = torch.sum(
        #     target_prob * (target_log_prob - pred_log_prob),
        #     dim=-1
        # )
        # # print("kl_per_atom.shape: ", kl_per_atom.shape)
        # # 4. mask & mean
        # kl_loss = kl_per_atom[valid_mask.squeeze(-1)].mean()
        # loss = kl_loss * 1
        # # # print("kl_loss: ", kl_loss)

        kl_loss = torch.tensor(0.0, device=fg_tokens.device)  # 转为张量
        loss = kl_loss * 1


        # target = sample[target_key]["fg_atom_relation_target"]
        # torch.set_printoptions(threshold=float('inf'))
        # print("masked_fg_atom_relation[1]: ", masked_fg_atom_relation[1])
        # print("sample[input_key]['fg_atom_relation'][1]: ", sample[input_key]['fg_atom_relation'][1])
        # print("target[1]: ", target[1])
        # sys.exit()


        # if masked_fg_atom_relation is not None:
        #     fg_atom_relation_target = (fg_atom_relation_target[masked_fg_atom_relation]-2).long()
        # # print("fg_atom_relation_target: ",fg_atom_relation_target)
        # # print("encoder_relation.shape: ",encoder_relation.shape)
        
        # masked_relation_loss = F.nll_loss(
        #     F.log_softmax(encoder_relation, dim=-1, dtype=torch.float32),
        #     fg_atom_relation_target,
        #     reduction="mean",
        # )
        # # print("masked_relation_loss: ",masked_relation_loss)
        # # sys.exit()
        # masked_pred = encoder_relation.argmax(dim=-1)
        # masked_hit = (masked_pred == fg_atom_relation_target).long().sum()
        # loss += masked_relation_loss * self.args.masked_relation_loss




        # print("fg_atom_dist_target.shape: ", fg_atom_dist_target.shape)
        # print("fg_atom_distance_pre.shape: ", fg_atom_distance_pre.shape)
        # print("fg_atom_dist_target: ", fg_atom_dist_target[:10])
        # print("fg_atom_distance_pre: ", fg_atom_distance_pre[:10])
        masked_fg_atom_dist_loss = F.smooth_l1_loss(
            fg_atom_distance_pre.view(-1).float(),
            fg_atom_dist_target.view(-1),
            reduction="mean",
            beta=1.0,
        )
        # print("masked_fg_atom_dist_loss: ", masked_fg_atom_dist_loss)
        loss = loss + masked_fg_atom_dist_loss * self.args.masked_fg_atom_dist_loss
        





        # print("atom_logits.shape: ", atom_logits.shape)
        # print("fg_logits.shape: ", fg_logits.shape)

        # print("fg_x_norm: ", fg_x_norm)
        # print("fg_atom_delta_pair_repr_norm: ", fg_atom_delta_pair_repr_norm)
        # print("sample[input_key][fg_tokens].shape: ", sample[input_key]['fg_tokens'].shape)
        # print("sample[input_key][fg_atom_relation].shape: ", sample[input_key]['fg_atom_relation'].shape)

        # print("fg_tokens: ", fg_tokens)


        # # batch内的官能团对比
        cl_loss = self.functional_group_contrastive_loss(
            fg_logits,
            fg_tokens,
            temperature=0.07
        )
        loss += cl_loss * 1
        # print("cl_loss: ", cl_loss)

        # # batch内的官能团层次对比
        # cl_loss = self.functional_group_contrastive_loss_hierarchical(
        #     fg_logits,
        #     fg_tokens,
        #     temperature=0.07
        # )
        # loss += cl_loss * 1
        # # print("cl_loss: ", cl_loss)




        # # batch内的官能团级别分子特征对比
        # cl_loss = self.fg_level_uniformity_loss(fg_logits[:,0,:])
        # loss += cl_loss * 1



        logging_output = {
            "sample_size": 1,
            "bsz": sample[input_key]["atom_tokens"].size(0),
            "atom_seq_len": sample[input_key]["atom_tokens"].size(1) * sample[input_key]["atom_tokens"].size(0),
            "fg_seq_len": sample[input_key]["fg_tokens"].size(1) * sample[input_key]["fg_tokens"].size(0),
            "kl_loss": kl_loss.data,
            "cl_loss": cl_loss.data,
            # "masked_relation_loss": masked_relation_loss.data,
            # "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
            "masked_fg_atom_dist_loss": masked_fg_atom_dist_loss.data,
        }

   


        # print("self.args.x_norm_loss: ",self.args.x_norm_loss)                                       # 0.01
        # print("self.args.delta_pair_repr_norm_loss: ",self.args.delta_pair_repr_norm_loss)           # 0.01


        if self.args.x_fg_norm_loss > 0 and fg_x_norm is not None:
            loss = loss + self.args.x_fg_norm_loss * fg_x_norm
            logging_output["fg_x_norm_loss"] = fg_x_norm.data


        if (self.args.delta_fg_atom_pair_repr_norm_loss > 0 and fg_atom_delta_pair_repr_norm is not None):
            loss += self.args.delta_fg_atom_pair_repr_norm_loss * fg_atom_delta_pair_repr_norm
            logging_output["fg_atom_delta_pair_repr_norm"] = fg_atom_delta_pair_repr_norm.data


        logging_output["loss"] = loss.data
        # print(logging_output)
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)


        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        metrics.log_scalar("kl_loss", kl_loss_sum / sample_size, sample_size, round=3)

        cl_loss_sum = sum(log.get("cl_loss", 0) for log in logging_outputs)
        metrics.log_scalar("cl_loss", cl_loss_sum / sample_size, sample_size, round=3)

        masked_fg_atom_dist_loss_sum = sum(log.get("masked_fg_atom_dist_loss", 0) for log in logging_outputs)
        metrics.log_scalar("masked_fg_atom_dist_loss", masked_fg_atom_dist_loss_sum / sample_size, sample_size, round=3)


        # masked_relation_loss_sum = sum(log.get("masked_relation_loss", 0) for log in logging_outputs)
        # metrics.log_scalar("masked_relation_loss", masked_relation_loss_sum / sample_size, sample_size, round=3)

        # masked_acc = sum(
        #     log.get("masked_token_hit", 0) for log in logging_outputs
        # ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        # metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)


        atom_seq_len = sum(log.get("atom_seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("atom_seq_len", atom_seq_len / bsz, 1, round=3)
        fg_seq_len = sum(log.get("fg_seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("fg_seq_len", fg_seq_len / bsz, 1, round=3)


        fg_x_norm_loss = sum(log.get("fg_x_norm_loss", 0) for log in logging_outputs)
        if fg_x_norm_loss > 0:
            metrics.log_scalar(
                "fg_x_norm_loss", fg_x_norm_loss / sample_size, sample_size, round=3
            )


        fg_atom_delta_pair_repr_norm = sum(
            log.get("fg_atom_delta_pair_repr_norm", 0) for log in logging_outputs
        )
        if fg_atom_delta_pair_repr_norm > 0:
            metrics.log_scalar(
                "fg_atom_delta_pair_repr_norm",
                fg_atom_delta_pair_repr_norm / sample_size,
                sample_size,
                round=3,
            )


    def fg_level_uniformity_loss(self, z, temperature=0.1):
        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, z.T) / temperature
        mask = torch.eye(z.size(0), device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        loss = torch.logsumexp(sim, dim=1).mean()
        return loss


    def functional_group_contrastive_loss_hierarchical(self,
        fg_logits: torch.Tensor,
        fg_tokens: torch.Tensor,
        temperature: float = 0.07,
    ):

        device = fg_logits.device
        B, K, D = fg_logits.shape

        # -------------------------------------------------
        # 1. 展平
        # -------------------------------------------------
        feat = fg_logits.reshape(-1, D)      # [B*K, D]
        type_id = fg_tokens.reshape(-1)      # [B*K]

        sample_id = (
            torch.arange(B, device=device)
            .unsqueeze(1)
            .repeat(1, K)
            .reshape(-1)
        )                                     # [B*K]

        # -------------------------------------------------
        # 2. 过滤无效官能团
        # -------------------------------------------------
        ignore_mask = torch.zeros_like(type_id, dtype=torch.bool)
        ignore_types = self.fg_dictionary.special_index()
        for t in ignore_types:
            ignore_mask |= (type_id == t)

        valid_fg_mask = ~ignore_mask

        feat = feat[valid_fg_mask]
        type_id = type_id[valid_fg_mask]
        sample_id = sample_id[valid_fg_mask]

        if feat.size(0) <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # -------------------------------------------------
        # 3. 特征归一化
        # -------------------------------------------------
        feat = F.normalize(feat, dim=-1)      # [N, D]
        N = feat.size(0)

        # -------------------------------------------------
        # 4. 相似度矩阵
        # -------------------------------------------------
        sim = torch.matmul(feat, feat.T) / temperature   # [N, N]

        # -------------------------------------------------
        # 5. 构造 mask
        # -------------------------------------------------
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        cross_sample = sample_id.unsqueeze(0) != sample_id.unsqueeze(1)

        # group_mask: [N, G]
        self.fg_group_matrix = self.fg_group_matrix.to(device)
        group_mask = self.fg_group_matrix[type_id]   # [N, G]  bool
        shared_group = torch.matmul(group_mask.float(),group_mask.float().T) > 0    # [N, N]，是否共享至少一个 group
        
        
        
        # 粗粒度     正样本：相同group     分母：全部
        pos_mask = (shared_group & cross_sample)
        valid_pair_mask = cross_sample & (~self_mask)
        # -------------------------------------------------
        # 6. 仅保留存在正 / 负样本的 query
        # -------------------------------------------------
        has_any_valid = valid_pair_mask.any(dim=1)
        sim_valid = sim[has_any_valid]
        pos_mask_valid = pos_mask[has_any_valid]
        pos_count = pos_mask_valid.sum(dim=1)
        valid_query = pos_count > 0
        if valid_query.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # -------------------------------------------------
        # 7. InfoNCE-style loss
        # -------------------------------------------------
        log_prob = sim_valid - torch.logsumexp(sim_valid, dim=1, keepdim=True)
        cl_loss_1 = -(
            (log_prob * pos_mask_valid).sum(dim=1) /
            pos_count.clamp(min=1)
        )[valid_query].mean()




        # 细粒度     正样本：相同官能团     分母：相同group
        pos_mask = (
            (type_id.unsqueeze(0) == type_id.unsqueeze(1)) &
            cross_sample
        )
        valid_pair_mask = (shared_group & cross_sample)
        # -------------------------------------------------
        # 6. 仅保留存在正 / 负样本的 query
        # -------------------------------------------------
        has_any_valid = valid_pair_mask.any(dim=1)
        sim_valid = sim[has_any_valid]
        pos_mask_valid = pos_mask[has_any_valid]
        pos_count = pos_mask_valid.sum(dim=1)
        valid_query = pos_count > 0
        if valid_query.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # -------------------------------------------------
        # 7. InfoNCE-style loss
        # -------------------------------------------------
        log_prob = sim_valid - torch.logsumexp(sim_valid, dim=1, keepdim=True)
        cl_loss_2 = -(
            (log_prob * pos_mask_valid).sum(dim=1) /
            pos_count.clamp(min=1)
        )[valid_query].mean()

        return cl_loss_1 + cl_loss_2
    



    def functional_group_contrastive_loss(self,
        fg_logits: torch.Tensor,
        fg_tokens: torch.Tensor,
        temperature: float = 0.07,
    ):
        """
        Batch 内官能团对比学习损失（跨样本）

        Args:
            fg_logits: Tensor [B, K, D]
                官能团特征
            fg_tokens: Tensor [B, K]
                官能团类型 id
            temperature: float
                对比学习温度系数
            ignore_types: iterable
                需要忽略的官能团类型（如 padding / special token）

        Returns:
            cl_loss: Tensor (scalar)
        """
        device = fg_logits.device
        B, K, D = fg_logits.shape

        # -------------------------------------------------
        # 1. 展平
        # -------------------------------------------------
        feat = fg_logits.reshape(-1, D)      # [B*K, D]
        type_id = fg_tokens.reshape(-1)      # [B*K]

        sample_id = (
            torch.arange(B, device=device)
            .unsqueeze(1)
            .repeat(1, K)
            .reshape(-1)
        )                                     # [B*K]

        # -------------------------------------------------
        # 2. 过滤无效官能团
        # -------------------------------------------------
        ignore_mask = torch.zeros_like(type_id, dtype=torch.bool)
        ignore_types = self.fg_dictionary.special_index()
        for t in ignore_types:
            ignore_mask |= (type_id == t)

        valid_fg_mask = ~ignore_mask

        feat = feat[valid_fg_mask]
        type_id = type_id[valid_fg_mask]
        sample_id = sample_id[valid_fg_mask]

        if feat.size(0) <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # -------------------------------------------------
        # 3. 特征归一化
        # -------------------------------------------------
        feat = F.normalize(feat, dim=-1)      # [N, D]
        N = feat.size(0)

        # -------------------------------------------------
        # 4. 相似度矩阵
        # -------------------------------------------------
        sim = torch.matmul(feat, feat.T) / temperature   # [N, N]

        # -------------------------------------------------
        # 5. 构造 mask
        # -------------------------------------------------
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        cross_sample = sample_id.unsqueeze(0) != sample_id.unsqueeze(1)

        pos_mask = (
            (type_id.unsqueeze(0) == type_id.unsqueeze(1)) &
            cross_sample
        )

        neg_mask = (
            (type_id.unsqueeze(0) != type_id.unsqueeze(1)) &
            cross_sample
        )

        valid_pair_mask = cross_sample & (~self_mask)

        # -------------------------------------------------
        # 6. 仅保留存在正 / 负样本的 query
        # -------------------------------------------------
        has_any_valid = valid_pair_mask.any(dim=1)

        sim_valid = sim[has_any_valid]
        pos_mask_valid = pos_mask[has_any_valid]

        pos_count = pos_mask_valid.sum(dim=1)
        valid_query = pos_count > 0

        if valid_query.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # -------------------------------------------------
        # 7. InfoNCE-style loss
        # -------------------------------------------------
        log_prob = sim_valid - torch.logsumexp(sim_valid, dim=1, keepdim=True)

        cl_loss = -(
            (log_prob * pos_mask_valid).sum(dim=1) /
            pos_count.clamp(min=1)
        )[valid_query].mean()

        return cl_loss




    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = sample["net_input"]["src_tokens"].ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        # 
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss


    def cal_atom_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        # print("dist_masked_tokens: ", dist_masked_tokens)   
        # tensor([[False, ..., False, ...,  True, False, False, False, False]])   被mask的地方，没几个True
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["atom_distance_target"][dist_masked_tokens]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)        
        # print("nb_masked_tokens: ", nb_masked_tokens)               # dist_masked_tokens 中 True 的数量
        masked_src_tokens = sample["net_input"]["atom_tokens"].ne(self.padding_idx)
        # print("masked_src_tokens: ", masked_src_tokens)        
        # tensor([[True, ..., True, ...,  True, False, False, False, False]])     是不padding_idx的地方，前面都是True

        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        # print("masked_src_tokens_expanded: ", masked_src_tokens_expanded)    # 维度：6个masked_src_tokens
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss






    def cal_fg_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["fg_distance_target"][
            dist_masked_tokens
        ]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = sample["net_input"]["fg_tokens"].ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        #
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss


    def cal_fg_atom_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        # print("dist_masked_tokens: ", dist_masked_tokens)   
        # tensor([[False, ..., False, ...,  True, False, False, False, False]])   被mask的地方，没几个True
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["fg_atom_distance_target"][dist_masked_tokens]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)        
        # print("nb_masked_tokens: ", nb_masked_tokens)               # dist_masked_tokens 中 True 的数量
        masked_src_tokens = sample["net_input"]["atom_tokens"].ne(self.padding_idx)
        # print("masked_src_tokens: ", masked_src_tokens)        
        # tensor([[True, ..., True, ...,  True, False, False, False, False]])     是不padding_idx的地方，前面都是True

        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        # print("masked_src_tokens_expanded.shape: ", masked_src_tokens_expanded.shape) 
        # print("masked_distance.shape: ", masked_distance.shape)    
        # print("masked_distance_target.shape: ", masked_distance_target.shape)    
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss






@register_loss("unimol_infer")
class UniMolInferLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.bos_idx = task.dictionary.bos()
        self.eos_idx = task.dictionary.eos()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_key = "net_input"
        target_key = "target"
        src_tokens = sample[input_key]["src_tokens"]
        token_mask = (src_tokens.ne(self.padding_idx) & src_tokens.ne(self.bos_idx) & src_tokens.ne(self.eos_idx))
        (
            encoder_rep,
            encoder_pair_rep,
        ) = model(**sample[input_key], features_only=True)
        sample_size = sample[input_key]["src_tokens"].size(0)
        encoder_rep_list = []
        encoder_pair_rep_list = []
        if 'pdb_id' in sample[target_key].keys():
            name_key = 'pdb_id'
        elif 'smi_name' in sample[target_key].keys():
            name_key = 'smi_name'
        else:
            raise NotImplementedError("No name key in the original data")

        for i in range(sample_size):  # rm padding bos eos token
            encoder_rep_list.append(encoder_rep[i][token_mask[i]].data.cpu().numpy())
            encoder_pair_rep_list.append(encoder_pair_rep[i][token_mask[i], :][:, token_mask[i]].data.cpu().numpy())
        logging_output = {
                "mol_repr_cls": encoder_rep[:, 0, :].data.cpu().numpy(),  # get cls token
                "atom_repr": encoder_rep_list,
                "pair_repr": encoder_pair_rep_list,
                "data_name": sample[target_key][name_key],
                "bsz": sample[input_key]["src_tokens"].size(0),
            }
        return 0, sample_size, logging_output








