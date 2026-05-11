# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
from unicore.data import Dictionary
from unicore.data import BaseWrapperDataset
from . import data_utils
import sys




class MaskFGAtomPointsDataset(BaseWrapperDataset):
    def __init__(
        self,
        token_dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        fg_token_dataset: torch.utils.data.Dataset,
        fg_coords_dataset: torch.utils.data.Dataset,
        fg_atom_dataset: torch.utils.data.Dataset,
        atom_vocab: Dictionary,
        fg_vocab: Dictionary,
        pad_idx: int,
        atom_mask_idx: int,
        fg_mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,                 # 15% 位置需要被 Mask
        leave_unmasked_prob: float = 0.1,        # 被 Mask 的位置中, 有 10% 不遮住（留原样）
        random_token_prob: float = 0.1,          # 被 Mask 的位置中, 有 10% 替换为随机 token, 剩下 80% 替换为 mask token
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = token_dataset
        self.coord_dataset = coord_dataset
        self.fg_token_dataset = fg_token_dataset
        self.fg_coords_dataset = fg_coords_dataset
        self.fg_atom_dataset = fg_atom_dataset
        self.atom_vocab = atom_vocab
        self.fg_vocab = fg_vocab
        self.pad_idx = pad_idx
        self.atom_mask_idx = atom_mask_idx
        self.fg_mask_idx = fg_mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        if random_token_prob > 0.0:
            weights = np.ones(len(self.atom_vocab))
            weights[atom_vocab.special_index()] = 0
            self.atom_weights = weights / weights.sum()
            weights = np.ones(len(self.fg_vocab))
            weights[fg_vocab.special_index()] = 0
            self.fg_weights = weights / weights.sum()             
            # print("self.atom_weights: ", self.atom_weights)
            # print("self.fg_weights: ", self.fg_weights)

        # print("self.noise_type: ", self.noise_type)    # self.noise_type:  uniform
        self.epoch = None
        if self.noise_type == "trunc_normal":
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            self.noise_f = lambda num_mask: 0.0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.fg_coords_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        # index = 60
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            fg_item = self.fg_token_dataset[index]
            fg_coords = self.fg_coords_dataset[index]
            fg_atom = self.fg_atom_dataset[index]

            fg_sz = len(fg_item)
            # don't allow empty sequence
            assert fg_sz > 0
            # decide elements to mask
            num_fg_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * fg_sz + np.random.rand()
                # self.mask_prob * fg_sz + np.random.rand()
            )
            num_fg_mask = max(1, num_fg_mask)

            mask_idcs = np.random.choice(fg_sz, num_fg_mask, replace=False)
            mask_idcs_final = []
            fg_mask = np.full(fg_sz, False)
            # 保证原子构成相同的官能团都被mask
            for mask_idc in mask_idcs: 
                for fg_id in fg_atom:
                    if fg_atom[fg_id] == fg_atom[mask_idc]:
                        mask_idcs_final.append(fg_id)
            fg_mask[mask_idcs_final] = True

            # 生成 targets 
            # print("self.pad_idx: ", self.pad_idx)
            ret["fg_targets"] = np.full(len(fg_mask), self.pad_idx)
            ret["fg_targets"][fg_mask] = fg_item[fg_mask]     # 被 mask 的位置要让模型预测原 token, 未 mask 的位置则 pad
            ret["fg_targets"] = torch.from_numpy(ret["fg_targets"]).long()

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob   # 随机 mask 和 不mask 的概率和
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = fg_mask & (np.random.rand(fg_sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    fg_unmask = rand_or_unmask
                    rand_fg_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    fg_unmask = None
                    rand_fg_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(fg_sz) < unmask_prob
                    fg_unmask = rand_or_unmask & decision
                    rand_fg_mask = rand_or_unmask & (~decision)
            else:
                fg_unmask = rand_fg_mask = None

            if fg_unmask is not None:
                fg_mask = fg_mask ^ fg_unmask         # 两个位置相同 → False， 两个位置不同 → True

            new_fg_item = np.copy(fg_item)
            new_fg_item[fg_mask] = self.fg_mask_idx
            num_fg_mask = fg_mask.astype(np.int32).sum()
            new_coord = np.copy(fg_coords)
            new_coord[fg_mask, :] += self.noise_f(num_fg_mask)

            if rand_fg_mask is not None:
                num_fg_rand = rand_fg_mask.sum()
                if num_fg_rand > 0:
                    new_fg_item[rand_fg_mask] = np.random.choice(
                        len(self.fg_vocab),
                        num_fg_rand,
                        p=self.fg_weights,
                    )
            ret["fgs"] = torch.from_numpy(new_fg_item).long()
            ret["fg_coordinates"] = torch.from_numpy(new_coord).float()
            # print("ret: ", ret)
            # print("fg_item: ", fg_item)
            # print("fg_coords: ", fg_coords)


            # print("rand_mask: ", rand_mask)


            # print("mask: ", mask)   # [False False False True False True False False False False False True ... False]
            # print("fg_atom: ", fg_atom) 

            len_atom = len(item)
            atom_must_mask = get_atom_mask(fg_mask, fg_atom, len_atom)
            # print("len_atom: ", len_atom)
            # print("fg_mask: ", fg_mask)
            # print("fg_atom: ", fg_atom)
            # print("atom_must_mask: ", atom_must_mask)            
            # sys.exit()            






            # 原子
            sz = len(item)
            # don't allow empty sequence
            assert sz > 0
            # decide elements to mask
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )
            num_mask = max(1, num_mask)

            mask_idcs = np.random.choice(sz, num_mask, replace=False)
            mask = np.full(sz, False)
            mask[mask_idcs] = True
            mask = atom_must_mask | mask
 
            # 生成 targets 
            # print("self.pad_idx: ", self.pad_idx)
            ret["targets"] = np.full(len(mask), self.pad_idx)
            ret["targets"][mask] = item[mask]     # 被 mask 的位置要让模型预测原 token, 未 mask 的位置则 pad
            ret["targets"] = torch.from_numpy(ret["targets"]).long()

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob   # 随机 mask 和 不mask 的概率和
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
                    # print("rand_mask: ",rand_mask)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask
            mask = atom_must_mask | mask 
            # print("atom_must_mask: ", atom_must_mask)
            # print("mask: ", mask)

            new_item = np.copy(item)
            new_item[mask] = self.atom_mask_idx
            # print("self.atom_mask_idx: ", self.atom_mask_idx)    # 30
            # sys.exit()
            num_mask = mask.astype(np.int32).sum()
            new_coord = np.copy(coord)
            new_coord[mask, :] += self.noise_f(num_mask)

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.atom_vocab),
                        num_rand,
                        p=self.atom_weights,
                    )
            ret["atoms"] = torch.from_numpy(new_item).long()
            ret["coordinates"] = torch.from_numpy(new_coord).float()

            # print("ret: ", ret)
            # print("fg_atom: ", fg_atom)
            # sys.exit() 

            return ret





class MaskAtomFGPointsDataset(BaseWrapperDataset):
    def __init__(
        self,
        token_dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        fg_token_dataset: torch.utils.data.Dataset,
        fg_coords_dataset: torch.utils.data.Dataset,
        fg_atom_dataset: torch.utils.data.Dataset,
        atom_vocab: Dictionary,
        fg_vocab: Dictionary,
        pad_idx: int,
        atom_mask_idx: int,
        fg_mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,                 # 15% 位置需要被 Mask
        leave_unmasked_prob: float = 0.1,        # 被 Mask 的位置中, 有 10% 不遮住（留原样）
        random_token_prob: float = 0.1,          # 被 Mask 的位置中, 有 10% 替换为随机 token, 剩下 80% 替换为 mask token
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = token_dataset
        self.coord_dataset = coord_dataset
        self.fg_token_dataset = fg_token_dataset
        self.fg_coords_dataset = fg_coords_dataset
        self.fg_atom_dataset = fg_atom_dataset
        self.atom_vocab = atom_vocab
        self.fg_vocab = fg_vocab
        self.pad_idx = pad_idx
        self.atom_mask_idx = atom_mask_idx
        self.fg_mask_idx = fg_mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        if random_token_prob > 0.0:
            weights = np.ones(len(self.atom_vocab))
            weights[atom_vocab.special_index()] = 0
            self.atom_weights = weights / weights.sum()
            weights = np.ones(len(self.fg_vocab))
            weights[fg_vocab.special_index()] = 0
            self.fg_weights = weights / weights.sum()             
            # print("self.atom_weights: ", self.atom_weights)
            # print("self.fg_weights: ", self.fg_weights)

        # print("self.noise_type: ", self.noise_type)    # self.noise_type:  uniform
        self.epoch = None
        if self.noise_type == "trunc_normal":
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            self.noise_f = lambda num_mask: 0.0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.fg_coords_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            fg_item = self.fg_token_dataset[index]
            fg_atom = self.fg_atom_dataset[index]
            fg_coords = self.fg_coords_dataset[index]


            sz = len(item)                          # 原子数量
            # don't allow empty sequence
            assert sz > 0
            # decide elements to mask
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )

            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask = np.full(sz, False)
            mask[mask_idc] = True

            # 生成 targets 
            # print("self.pad_idx: ", self.pad_idx)
            ret["targets"] = np.full(len(mask), self.pad_idx)
            ret["targets"][mask] = item[mask]     # 被 mask 的位置要让模型预测原 token, 未 mask 的位置则 pad
            ret["targets"] = torch.from_numpy(ret["targets"]).long()

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob   # 随机 mask 和 不mask 的概率和
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            new_item = np.copy(item)
            new_item[mask] = self.atom_mask_idx
            # print("self.atom_mask_idx: ", self.atom_mask_idx)    # 30
            # sys.exit()
            num_mask = mask.astype(np.int32).sum()
            new_coord = np.copy(coord)
            new_coord[mask, :] += self.noise_f(num_mask)

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.atom_vocab),
                        num_rand,
                        p=self.atom_weights,
                    )
            ret["atoms"] = torch.from_numpy(new_item).long()
            ret["coordinates"] = torch.from_numpy(new_coord).float()

            # print("rand_mask: ", rand_mask)


            # print("mask: ", mask)   # [False False False True False True False False False False False True ... False]
            # print("fg_atom: ", fg_atom) 

            fg_mask = get_fg_mask(mask, fg_atom)
            
            if any(fg_mask):      # 如果正常
                # print("存在fg")
                new_fg_item = np.copy(fg_item)
                new_fg_item[fg_mask] = self.fg_mask_idx
                num_fg_rand = int(np.ceil(sum(fg_mask) * 0.1 / (0.1+0.8)))
                if num_fg_rand > 0:
                    rand_fg_mask = keep_random_true(fg_mask, num_fg_rand)
                    # 随机把num_fg_rand个是True的地方保留，其他的都变成false
                    new_fg_item[rand_fg_mask] = np.random.choice(
                        len(self.fg_vocab),
                        num_fg_rand,
                        p=self.fg_weights,
                    )

                # 基于原子的噪声坐标，得到官能团的噪声坐标
                noisy_fg_coords = np.copy(fg_coords)
                for fg_id, is_mask in enumerate(fg_mask):
                    if is_mask:
                        coords = np.array([list(new_coord[i]) for i in fg_atom[fg_id]])
                        fg_center = coords.mean(axis=0)  # 平均坐标
                        noisy_fg_coords[fg_id] = fg_center

            else:        # 如果全是False,那么随机选择一定数量的官能团为True
                # print("不存在fg")
                n = len(fg_mask)
                ratio = 0.15*0.9
                k = max(1, int(np.ceil(n * ratio)))
                idx = np.random.choice(n, k, replace=False)
                for i in idx:
                    fg_mask[i] = True
                
                new_fg_item = np.copy(fg_item)
                new_fg_item[fg_mask] = self.fg_mask_idx                
                num_fg_rand = int(np.ceil(sum(fg_mask) * 0.1 / (0.1+0.8)))
                if num_fg_rand > 0:
                    rand_fg_mask = keep_random_true(fg_mask, num_fg_rand)
                    # 随机把num_fg_rand个是True的地方保留，其他的都变成false
                    new_fg_item[rand_fg_mask] = np.random.choice(
                        len(self.fg_vocab),
                        num_fg_rand,
                        p=self.fg_weights,
                    )
                # 随机得到官能团的噪声坐标
                noisy_fg_coords = np.copy(fg_coords)
                noisy_fg_coords[fg_mask, :] += self.noise_f(sum(fg_mask))

            ret["fg_targets"] = np.full(len(fg_mask), self.pad_idx)
            ret["fg_targets"][fg_mask] = fg_item[fg_mask]     # 被 mask 的位置要让模型预测原 token, 未 mask 的位置则 pad
            ret["fg_targets"] = torch.from_numpy(ret["fg_targets"]).long()

            ret["fgs"] = torch.from_numpy(new_fg_item).long()
            ret["fg_coordinates"] = torch.from_numpy(noisy_fg_coords).float()

            ret["fg_coords"] = torch.from_numpy(fg_coords).float()
            return ret



def get_fg_mask(mask, fg_atom):
    fg_mask = [False] * len(fg_atom)
    masked_atoms = set(np.where(mask)[0])  # 得到所有被 mask 的原子ID


    for fg_id, atom_ids in fg_atom.items():
        # 判断这个官能团是否包含任何被mask的原子
        if masked_atoms.intersection(atom_ids):
            fg_mask[fg_id] = True
        else:
            fg_mask[fg_id] = False

    return fg_mask


def get_atom_mask(fg_mask, fg_atom, len_atom):
    atom_mask = [False] * len_atom
    masked_fg = set(np.where(fg_mask)[0])  # 得到所有被 mask 的fg ID

    for fg_id, atom_ids in fg_atom.items():
        if fg_id in masked_fg:
            for atom_id in atom_ids:
                atom_mask[atom_id] = True

    return atom_mask




def keep_random_true(fg_mask, num_fg_rand):
    fg_mask = list(fg_mask)  # 保证是 list
    true_indices = [i for i, v in enumerate(fg_mask) if v]

    # 如果原本 True 的数量 <= num_fg_rand，直接返回，不需要操作
    if len(true_indices) <= num_fg_rand:
        return fg_mask

    # 随机选 num_fg_rand 个要保留的 True 位置
    keep_idx = set(np.random.choice(true_indices, num_fg_rand, replace=False))

    # 构造新的 mask：只保留选中的 True
    new_fg_mask = [(i in keep_idx) for i in range(len(fg_mask))]
    return new_fg_mask





class MaskPointsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,                 # 15% 位置需要被 Mask
        leave_unmasked_prob: float = 0.1,        # 被 Mask 的位置中, 有 10% 不遮住（留原样）
        random_token_prob: float = 0.1,          # 被 Mask 的位置中, 有 10% 替换为随机 token, 剩下 80% 替换为 mask token
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.coord_dataset = coord_dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()
            # print("self.weights: ", self.weights)

        # print("self.noise_type: ", self.noise_type)    # self.noise_type:  uniform
        self.epoch = None
        if self.noise_type == "trunc_normal":
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            self.noise_f = lambda num_mask: 0.0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            # print("item: ", item)     # item:  tensor([4, 6, 4, 6, 4, 4, 4, 5, 4, 6, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4])
            # print("coord: ", coord)   # 一个构象的坐标

            sz = len(item)                          # 原子数量
            # don't allow empty sequence
            assert sz > 0
            # decide elements to mask
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz
                + np.random.rand()
            )

            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask = np.full(sz, False)
            mask[mask_idc] = True

            # 生成 targets 
            # print("self.pad_idx: ", self.pad_idx)
            ret["targets"] = np.full(len(mask), self.pad_idx)
            ret["targets"][mask] = item[mask]     # 被 mask 的位置要让模型预测原 token, 未 mask 的位置则 pad
            ret["targets"] = torch.from_numpy(ret["targets"]).long()

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob   # 随机 mask 和 不mask 的概率和
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            # print("self.mask_idx: ", self.mask_idx)    # 30
            # sys.exit()
            num_mask = mask.astype(np.int32).sum()
            new_coord = np.copy(coord)
            new_coord[mask, :] += self.noise_f(num_mask)

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )
            ret["atoms"] = torch.from_numpy(new_item).long()
            ret["coordinates"] = torch.from_numpy(new_coord).float()
            
            return ret


class MaskPointsPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        residue_dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.coord_dataset = coord_dataset
        self.residue_dataset = residue_dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()

        self.epoch = None
        if self.noise_type == "trunc_normal":
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            self.noise_f = lambda num_mask: 0.0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]           
            coord = self.coord_dataset[index]    
            sz = len(item)
            # don't allow empty sequence
            assert sz > 0

            # mask on the level of residues
            residue = self.residue_dataset[index]
            res_list = list(set(residue))
            res_sz = len(res_list)

            # decide elements to mask
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * res_sz
                + np.random.rand()
            )
            mask_res = np.random.choice(res_list, num_mask, replace=False).tolist()
            mask = np.isin(residue, mask_res)

            ret["targets"] = np.full(len(mask), self.pad_idx)
            ret["targets"][mask] = item[mask]
            ret["targets"] = torch.from_numpy(ret["targets"]).long()
            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx

            num_mask = mask.astype(np.int32).sum()
            new_coord = np.copy(coord)
            new_coord[mask, :] += self.noise_f(num_mask)

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )
            ret["atoms"] = torch.from_numpy(new_item).long()
            ret["coordinates"] = torch.from_numpy(new_coord).float()
            return ret










