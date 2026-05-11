# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys


import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    RawArrayDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    Add2DConformerDataset,
    LMDBDataset,
    TTADataset,
    DoNotAdd2DConformerDataset,
    AddFunctionGroupDataset,
    MaskAtomFGPointsDataset,
    MaskFGAtomPointsDataset,
    NormalizeWithRefDataset,
    CrossDistanceDataset,
    RightPadDatasetCross2D,
    CrossEdgeTypeDataset,
)
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("unimol")
class UniMolTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
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
            "--atom-dict-name",
            default="atom_dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--fg-dict-name",
            default="fg_dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        atom_dictionary = Dictionary.load(os.path.join(args.data, args.atom_dict_name))  
        fg_dictionary = Dictionary.load(os.path.join(args.data, args.fg_dict_name)) 
        # print(args.data, args.dict_name)
        # /data/workspace/tanhaojiang/AI4Sci/Uni-Mol-main/unimol/example_data/molecule/      dict.txt
        logger.info("atom dictionary: {} types".format(len(atom_dictionary)))
        logger.info("fg dictionary: {} types".format(len(fg_dictionary)))
        return cls(args, atom_dictionary, fg_dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")

        raw_dataset = LMDBDataset(split_path)

        def one_dataset(raw_dataset, coord_seed, mask_seed):
            if self.args.mode =='train':
                smi_dataset = KeyDataset(raw_dataset, "smi")
                dataset = ConformerSampleDataset(
                    raw_dataset, coord_seed, "atoms", "coordinates", "func_groups", "fg_atom"
                )
                dataset = AtomTypeDataset(raw_dataset, dataset)


            elif self.args.mode == 'infer':
                dataset = TTADataset(
                    raw_dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
                )
                dataset = AtomTypeDataset(dataset, dataset)
                smi_dataset = KeyDataset(dataset, "smi")
            dataset = RemoveHydrogenDataset(
                dataset,
                "atoms",
                "coordinates",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
            )



            dataset = AddFunctionGroupDataset(
                dataset, "smi", "atoms", "coordinates", "func_groups", "fg_atom", self.args.max_atoms
            )  
            # print("dataset[1]: ", dataset[100])
            # sys.exit()           
            # dataset = RemoveHydrogenDataset(
            #     dataset,
            #     "atoms",
            #     "coordinates",
            #     self.args.remove_hydrogen,
            #     self.args.remove_polar_hydrogen,
            # )

            dataset = CroppingDataset(
                dataset, self.seed, "atoms", "coords", self.args.max_atoms
            )
            dataset = NormalizeDataset(dataset, "coords", normalize_coord=True)
            dataset = NormalizeDataset(dataset, "fg_coords", normalize_coord=True)
            


            token_dataset = KeyDataset(dataset, "atoms")
            token_dataset = TokenizeDataset(
                token_dataset, self.atom_dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(dataset, "coords")
            coord_dataset = FromNumpyDataset(coord_dataset)

            fg_token_dataset = KeyDataset(dataset, "func_groups")
            fg_token_dataset = TokenizeDataset(
                fg_token_dataset, self.fg_dictionary, max_seq_len=self.args.max_seq_len
            )            
            fg_coords_dataset = KeyDataset(dataset, "fg_coords")
            fg_coords_dataset = FromNumpyDataset(fg_coords_dataset)

            fg_atom_dataset = KeyDataset(dataset, "fg_atom")
            

            fg_atom_matrix_dataset = KeyDataset(dataset, "fg_atom_matrix_padding")
            fg_atom_matrix_dataset = FromNumpyDataset(fg_atom_matrix_dataset)

            fg_atom_matrix_target_dataset = KeyDataset(dataset, "fg_atom_matrix_target")
            fg_atom_matrix_target_dataset = FromNumpyDataset(fg_atom_matrix_target_dataset)



            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)


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
            fg_coords_dataset = PrependAndAppend(fg_coords_dataset, 0.0, 0.0)
            fg_distance_dataset = DistanceDataset(fg_coords_dataset)
            fg_edge_type = EdgeTypeDataset(src_fg_dataset, len(self.fg_dictionary))


            # 针对官能团-原子
            fg_atom_distance_dataset = CrossDistanceDataset(fg_coords_dataset, coord_dataset)
            fg_atom_edge_type = CrossEdgeTypeDataset(src_fg_dataset, src_dataset, len(self.atom_dictionary))
            
            return {
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
                    fg_coords_dataset,
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
                "fg_atom_relation": RightPadDatasetCross2D(
                    fg_atom_matrix_dataset,
                    pad_idx=0,
                ),           
                "smi_dataset": RawArrayDataset(smi_dataset),
            },{
                "fg_atom_relation_target": RightPadDatasetCross2D(
                    fg_atom_matrix_target_dataset,
                    pad_idx=0,
                ),                 
            }

        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {"net_input": net_input, "target": target, "smi_name": net_input["smi_dataset"],}
        
        
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset



    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model

    def disable_shuffling(self) -> bool:
        return True
    
