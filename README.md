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
├── unimol_pretrained/
│   └── mol_pre_no_h_220816.pt
```

---

# Pretraining Dataset

The pretraining dataset should be downloaded from the official [Uni-Mol Dataset Repository](https://github.com/dptech-corp/Uni-Mol).

After downloading, place the dataset under:

```bash
data/pretrain/
```

Example:

```bash
data/
├── pretrain/
│   ├── train.lmdb
│   ├── valid.lmdb
│   └── dict.txt
```

---

# Data Preprocessing

Run the preprocessing script before training:

```bash
python XXX.py
```

Please replace `XXX.py` with the actual preprocessing script.

---

# GeoFG Pretraining

Example command for GeoFG pretraining:

```bash
bash scripts/pretrain.sh
```

or

```bash
python train.py \
    --task pretrain \
    --config config/pretrain.yaml
```

---

# GeoFG Finetuning

Example command for downstream finetuning:

```bash
bash scripts/finetune.sh
```

or

```bash
python finetune.py \
    --task classification \
    --config config/finetune.yaml
```

---

# Project Structure

```bash
.
├── data/
├── scripts/
├── config/
├── unimol/
├── fg_unimol/
├── environment.yml
├── README.md
└── train.py
```

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
