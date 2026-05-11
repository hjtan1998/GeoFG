# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from collections import defaultdict
import sys
import copy

class AddFunctionGroupDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates, func_groups, fg_atom, max_atoms):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.func_groups = func_groups
        self.fg_atom = fg_atom
        self.max_atoms = max_atoms
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        # print("index: ", index)
        atoms = np.array(self.dataset[index][self.atoms])
        # print("atoms_ini: ", atoms)
        # sys.exit()
        assert len(atoms) > 0
        smi = self.dataset[index][self.smi]
        coordinates = self.dataset[index][self.coordinates]
        old_func_groups = self.dataset[index][self.func_groups]
        old_fg_atom = self.dataset[index][self.fg_atom]

        # 获取官能团类型、坐标、以及与原子的关联
        mol = Chem.MolFromSmiles(smi)
        atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # if 'H' in atom_list:
        #     print("有 H！！！")
        #     mol = Chem.RemoveHs(mol)    # 不能移除「隐式 H（implicit H）」
        #     atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
            
        # print("atom_list: ", atom_list)
        # sys.exit()

        fg_coords = []
        func_groups = []
        fg_atom = {}
        new_idx = 0
        for idx, fg_type in enumerate(old_func_groups):
            atom_ids = old_fg_atom[idx]
            # # # 获取官能团原子坐标
            # print("atom_ids: ",atom_ids)
            # print("coordinates len: ", len(coordinates))
            if 'H' in atom_list:
                # print("atoms_ini: ", atoms)
                # print("atom_list: ", atom_list)
                # print("atom_ids: ", atom_ids)
                # print("len(coordinates): ", len(coordinates))
                mapping = {}
                heavy_idx = 0
                for i, atom in enumerate(atom_list):
                    if atom != 'H':
                        mapping[i] = heavy_idx
                        heavy_idx += 1
                coords = np.array([list(coordinates[mapping[i]]) for i in atom_ids if i in mapping])
                fg_center = coords.mean(axis=0)  # 平均坐标
                fg_coords.append(fg_center)  
                func_groups.append(fg_type)            
            else:
                if max(atom_ids) < self.max_atoms:
                    coords = np.array([list(coordinates[i]) for i in atom_ids])
                    fg_center = coords.mean(axis=0)  # 平均坐标
                    fg_coords.append(fg_center)
                    func_groups.append(fg_type)
                    fg_atom[new_idx] = old_fg_atom[idx]
                    new_idx += 1        
        fg_coords = np.array(fg_coords)



        # 四边是1，没有关联的官能团-原子是2，有关联的是3
        fg_atom_matrix_padding = np.ones((len(func_groups)+2, len(atoms)+2))
        fg_atom_matrix_padding[1:-1,1:-1] = 2

        if 'H' in atom_list: 
            for idx in fg_atom:
                for atom_idx in fg_atom[idx]: 
                    if atom_idx in mapping:
                        fg_atom_matrix_padding[idx+1, mapping[atom_idx]+1] = 3              
        else:
            for idx in fg_atom:
                atom_ids = fg_atom[idx]
                if max(atom_ids) < self.max_atoms:
                    for atom_idx in atom_ids: 
                        fg_atom_matrix_padding[idx+1, atom_idx+1] = 3
        
        
        # print("fg_atom: ", fg_atom)
        # print("fg_atom_matrix_padding: ",fg_atom_matrix_padding)
        # sys.exit()

        if len(func_groups)==0:
            func_groups = ['NoFuncGroups', 'NoFuncGroups']
            fg_coords = np.array([[0,0,0],[0,0,0]])



        # 自监督掩码    随机mask
        # 1. 初始化 target
        fg_atom_matrix_target = np.zeros_like(fg_atom_matrix_padding)
        # 2. 找到所有值为 3 和 2 的位置
        pos_3 = np.argwhere(fg_atom_matrix_padding == 3)
        pos_2 = np.argwhere(fg_atom_matrix_padding == 2)
        # 3. 从 3 中随机选 10%
        num_mask = max(1, int(0.2 * len(pos_3)))  # 至少 mask 1 个
        # mask_3_idx = np.random.choice(len(pos_3), num_mask, replace=False)
        if len(pos_3) == 0 or num_mask == 0:
            mask_3_idx = np.array([], dtype=np.int64)
        else:
            num_mask = min(num_mask, len(pos_3))
            mask_3_idx = np.random.choice(len(pos_3), num_mask, replace=False)
        mask_3_pos = pos_3[mask_3_idx]        
        
        # 4. 从 2 中随机选相同数量
        if len(pos_2) == 0 or num_mask == 0:
            mask_2_idx = np.array([], dtype=np.int64)
        else:
            num_mask = min(num_mask, len(pos_2))
            mask_2_idx = np.random.choice(len(pos_2), num_mask, replace=False)
        mask_2_pos = pos_2[mask_2_idx]
        # 5. 记录 target（掩码前的原始值）
        for i, j in np.vstack([mask_3_pos, mask_2_pos]):
            fg_atom_matrix_target[i, j] = fg_atom_matrix_padding[i, j]
        # 6. 将这些位置置为掩码 4
        fg_atom_matrix_padding_with_mask = copy.deepcopy(fg_atom_matrix_padding)
        fg_atom_matrix_padding_with_mask[mask_3_pos[:, 0], mask_3_pos[:, 1]] = 4
        fg_atom_matrix_padding_with_mask[mask_2_pos[:, 0], mask_2_pos[:, 1]] = 4



        # # # 自监督掩码    按行mask
        # M, N = fg_atom_matrix_padding.shape

        # # 1. 初始化 target
        # fg_atom_matrix_target = np.zeros_like(fg_atom_matrix_padding)

        # # 2. 找到「有效官能团行」（至少有一个非 padding=1 的位置）
        # valid_fg_rows = np.where(
        #     np.any(fg_atom_matrix_padding != 1, axis=1)
        # )[0]

        # # 如果没有可 mask 的官能团，直接返回
        # if len(valid_fg_rows) == 0:
        #     fg_atom_matrix_padding_with_mask = fg_atom_matrix_padding.copy()
        # else:
        #     # 3. 随机选 20% 的官能团（整行）
        #     num_mask_fg = max(1, int(0.2 * len(valid_fg_rows)))
        #     num_mask_fg = min(num_mask_fg, len(valid_fg_rows))
        #     mask_fg_rows = np.random.choice(valid_fg_rows, num_mask_fg, replace=False)

        #     # 4. 记录 target（仅监督被 mask 的官能团行，padding 位置仍为 0）
        #     fg_atom_matrix_target[mask_fg_rows, :] = fg_atom_matrix_padding[mask_fg_rows, :]
        #     fg_atom_matrix_target[mask_fg_rows, 0] = 0
        #     fg_atom_matrix_target[mask_fg_rows, -1] = 0
        #     # print("fg_atom_matrix_target: ", fg_atom_matrix_target)
        #     # sys.exit()

        #     # 5. 执行整行 mask（但不改 padding=1 的位置）
        #     fg_atom_matrix_padding_with_mask = fg_atom_matrix_padding.copy()

        #     for r in mask_fg_rows:
        #         non_padding_cols = fg_atom_matrix_padding[r] != 1
        #         fg_atom_matrix_padding_with_mask[r, non_padding_cols] = 4

        #     # print("fg_atom_matrix_padding: ", fg_atom_matrix_padding)
        #     # print("fg_atom_matrix_padding_with_mask: ", fg_atom_matrix_padding_with_mask)
        #     # sys.exit()










        # print("fg_atom_matrix_padding: ", fg_atom_matrix_padding)
        # print("fg_atom_matrix_padding_with_mask: ", fg_atom_matrix_padding_with_mask)
        # print("fg_atom_matrix_target: ", fg_atom_matrix_target)







        return {
                "smi": smi, 
                "atoms": atoms, 
                "coords": coordinates, 
                "func_groups": func_groups, 
                "fg_coords": fg_coords, 
                "fg_atom": fg_atom, 
                "fg_atom_matrix_padding": fg_atom_matrix_padding,
                "fg_atom_matrix_padding_with_mask": fg_atom_matrix_padding_with_mask,
                "fg_atom_matrix_target": fg_atom_matrix_target,
            }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)













def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates







