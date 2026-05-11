# GeoFG

This repository contains the official source code for the paper:

> Geometry-Aware Functional Group Modeling for 3D Molecular Representation Learning

---

# Environment Setup

We recommend using Conda to create the environment from the provided `environment.yml` file.

## Create Environment

```bash
conda env create -f environment.yml
```

## Activate Environment

```bash
conda activate alidiff
```

---

# Pretrained Model Preparation

Please download the Uni-Mol pretrained model and place it into the `data/` directory.

Example structure:

```bash
data/
├── mol_pre_no_h_220816.pt
```

---

# Pretraining Dataset

The pretraining dataset should be downloaded from the official [Uni-Mol Dataset Repository](https://github.com/dptech-corp/Uni-Mol).

After downloading, place the dataset under:

```bash
data/ligands/
```

Example:

```bash
data/
├── ligands/
│   ├── train.lmdb
│   ├── valid.lmdb
│   └── dict.txt
```

---

# Data Preprocessing

Run the preprocessing script before training:

```bash
python data/add_fg.py
```

---

# GeoFG Pretraining

Example command for GeoFG pretraining:

```bash
bash scripts/pretrain.sh
```


---

# GeoFG Finetuning

Example command for downstream finetuning:

```bash
bash scripts/run_all_finetune.sh
```


---



---

# Citation

If you find this work useful, please consider citing our paper.

```bibtex
@article{GeoFG,
  title={Geometry-Aware Functional Group Modeling for 3D Molecular Representation Learning},
  author={Anonymous Authors},
  journal={},
  year={2026}
}
```

---

# Acknowledgements

This project is built upon the Uni-Mol framework.  
We thank the authors of Uni-Mol for their open-source contributions.
