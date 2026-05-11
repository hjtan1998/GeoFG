# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class TTADataset(BaseWrapperDataset):
    # 这是一个 Test-Time Augmentation (TTA) 数据集包装器。
    # 用于在 推理 / 验证 / 测试阶段：
    # 对同一个分子（SMILES）生成 多个 conformers（3D 构象）
    # 将这些多次输入分别送入模型
    # 模型对同一个样本得到多个预测→ 最后取平均，提高鲁棒性（减少构象偏差）

    def __init__(self, dataset, seed, atoms, coordinates, func_groups, fg_atom, conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.func_groups = func_groups
        self.fg_atom = fg_atom
        self.conf_size = conf_size
        print("self.conf_size: ", self.conf_size)
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        smi = self.dataset[smi_idx]["smi"]
        target = self.dataset[smi_idx].get("target", None)
        func_groups = self.dataset[smi_idx][self.func_groups]
        fg_atom = self.dataset[smi_idx][self.fg_atom]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "smi": smi,
            "target": target,
            "func_groups": func_groups,
            "fg_atom": fg_atom,
        }
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
        conf_size=10,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        pocket_atoms = np.array(
            [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
        )
        pocket_coordinates = np.array(self.dataset[smi_idx][self.pocket_coordinates][0])
        if self.is_train:
            holo_coordinates = np.array(self.dataset[smi_idx][self.holo_coordinates][0])
            holo_pocket_coordinates = np.array(
                self.dataset[smi_idx][self.holo_pocket_coordinates][0]
            )
        else:
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates

        smi = self.dataset[smi_idx]["smi"]
        pocket = self.dataset[smi_idx]["pocket"]

        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
