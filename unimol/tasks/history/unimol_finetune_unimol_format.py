# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
    AddFunctionGroupDataset,
    MaskAtomFGPointsDataset,
    NormalizeWithRefDataset,
    CrossDistanceDataset,
    RightPadDatasetCross2D,
    CrossEdgeTypeDataset,
)

from unimol.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
}


@register_task("mol_finetune")
class UniMolFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )




    def __init__(self, args, atom_dictionary, fg_dictionary):
        super().__init__(args)
        self.atom_dictionary = atom_dictionary
        self.fg_dictionary = fg_dictionary
        self.seed = args.seed
        # add mask token
        self.atom_mask_idx = atom_dictionary.add_symbol("[MASK]", is_special=True)
        self.fg_mask_idx = fg_dictionary.add_symbol("[MASK]", is_special=True)
        
        
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]





    @classmethod
    def setup_task(cls, args, **kwargs):
        atom_dictionary = Dictionary.load("/data/workspace/tanhaojiang/mol_property/H2EMol/data/unimol_format-5m-1k/atom_dict.txt")  
        fg_dictionary = Dictionary.load("/data/workspace/tanhaojiang/mol_property/H2EMol/data/unimol_format-5m-1k/fg_dict.txt") 
        # print(args.data, args.dict_name)
        # /data/workspace/tanhaojiang/AI4Sci/Uni-Mol-main/unimol/example_data/molecule/      dict.txt
        logger.info("atom dictionary: {} types".format(len(atom_dictionary)))
        logger.info("fg dictionary: {} types".format(len(fg_dictionary)))
        return cls(args, atom_dictionary, fg_dictionary)


    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        if split == "train":
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")                
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
            # print("split: ", split)
            # print(dataset[0].keys())
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")            
            # print("split: ", split)
            # print(dataset[0].keys())
        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )       
        # print("dataset[0]  1: ", dataset[0])

        dataset = AddFunctionGroupDataset(
                dataset, "smi", "atoms", "coords"
            )
        # print("dataset[0]: ", dataset[0])


        # dataset = CroppingDataset(
        #     dataset, self.seed, "atoms", "coords", self.args.max_atoms
        # )

        dataset = NormalizeDataset(dataset, "coords", normalize_coord=True)
        dataset = NormalizeDataset(dataset, "fg_coords", normalize_coord=True)
            
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)





        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(
            token_dataset, self.atom_dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coords")

        
        fg_token_dataset = KeyDataset(dataset, "func_groups")
        fg_token_dataset = TokenizeDataset(
            fg_token_dataset, self.fg_dictionary, max_seq_len=self.args.max_seq_len
        )
        fg_coords_dataset = KeyDataset(dataset, "fg_coords")


        coord_dataset = FromNumpyDataset(coord_dataset)
        fg_coords_dataset = FromNumpyDataset(fg_coords_dataset)




        # 针对原子
        src_dataset = PrependAndAppend(
            token_dataset, self.atom_dictionary.bos(), self.atom_dictionary.eos()
        )
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        edge_type = EdgeTypeDataset(src_dataset, len(self.atom_dictionary))
        
        # 针对官能团
        src_fg_dataset = PrependAndAppend(
            fg_token_dataset, self.fg_dictionary.bos(), self.fg_dictionary.eos()
        )
        fg_coord_dataset = PrependAndAppend(fg_coords_dataset, 0.0, 0.0)
        fg_distance_dataset = DistanceDataset(fg_coord_dataset)
        fg_edge_type = EdgeTypeDataset(src_fg_dataset, len(self.fg_dictionary))

        # 针对官能团-原子
        fg_atom_distance_dataset = CrossDistanceDataset(fg_coord_dataset, coord_dataset)
        fg_atom_edge_type = CrossEdgeTypeDataset(src_fg_dataset, src_dataset, len(self.atom_dictionary))








        # print("len src_dataset:", len(src_dataset))
        # print("len coord_dataset:", len(coord_dataset))
        # print("len distance_dataset:", len(distance_dataset))
        # print("len edge_type:", len(edge_type))
        # print("len src_fg_dataset:", len(src_fg_dataset))
        # print("len fg_coord_dataset:", len(fg_coord_dataset))
        # print("len fg_distance_dataset:", len(fg_distance_dataset))
        # print("len fg_edge_type:", len(fg_edge_type))
        # print("len fg_atom_distance_dataset:", len(fg_atom_distance_dataset))
        # print("len fg_atom_edge_type:", len(fg_atom_edge_type))
        # print("len tgt_dataset:", len(tgt_dataset))
        # print("len smi_dataset:", len(smi_dataset))

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "atom_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.atom_dictionary.pad(),
                    ),
                    "atom_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "atom_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "atom_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "fg_tokens": RightPadDataset(
                        src_fg_dataset,
                        pad_idx=self.fg_dictionary.pad(),
                    ),
                    "fg_coord": RightPadDatasetCoord(
                        fg_coord_dataset,
                        pad_idx=0,
                    ),
                    "fg_distance": RightPadDataset2D(
                        fg_distance_dataset,
                        pad_idx=0,
                    ),
                    "fg_edge_type": RightPadDataset2D(
                        fg_edge_type,
                        pad_idx=0,
                    ),
                    "fg_atom_distance": RightPadDatasetCross2D(
                        fg_atom_distance_dataset,
                        pad_idx=0,
                    ),
                    "fg_atom_edge_type": RightPadDatasetCross2D(
                        fg_atom_edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
