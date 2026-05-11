import lmdb
import pickle
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem
from collections import defaultdict
import sys


from multiprocessing import Pool


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates



# =====================================================
# 配置
# =====================================================
SRC_LMDB = "/home/aizoo/data/workspace/tanhaojiang/mol_property/unimol_data/ligands/valid.lmdb"
DST_LMDB = "/home/aizoo/data/workspace/tanhaojiang/mol_property/unimol_data/ligands_fg/valid.lmdb"

FDEF_PATH = "BaseFeatures.fdef"
CHECKPOINT = "fg_checkpoint_valid.txt"

COMMIT_INTERVAL = 500    # 每 500 条提交一次（更安全）
MAP_SIZE = 5 * 1024**4   # 5 TB

# =====================================================
# RDKit Feature Factory
# =====================================================
fdef = ChemicalFeatures.BuildFeatureFactory(FDEF_PATH)





def process_one(idx):

    # 每个进程自己初始化
    src_env = lmdb.open(
        SRC_LMDB,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    fdef = ChemicalFeatures.BuildFeatureFactory(FDEF_PATH)

    with src_env.begin() as txn:
        key = str(idx).encode("ascii")
        raw = txn.get(key)
        if raw is None:
            return None

        data = pickle.loads(raw)

    smi = data["smi"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    func_groups = []
    fg_atom = defaultdict(list)

    features = fdef.GetFeaturesForMol(mol)
    for fidx, feat in enumerate(features):
        fg_atom[fidx] = list(feat.GetAtomIds())
        func_groups.append(feat.GetType())

    coordinates = data["coordinates"]
    coordinates.append(np.array(smi2_2Dcoords(smi)))

    data.update({
        "func_groups": func_groups,
        "fg_atom": dict(fg_atom),
        "coordinates": coordinates
    })

    return idx, data








NUM_WORKERS = 16      # 按 CPU 调
COMMIT_INTERVAL = 500

dst_env = lmdb.open(
    DST_LMDB,
    subdir=False,
    map_size=MAP_SIZE,
    lock=True,
)

# 读取长度
src_env = lmdb.open(SRC_LMDB, subdir=False, readonly=True, lock=False)
with src_env.begin() as txn:
    length = txn.stat()["entries"]
src_env.close()

# checkpoint
start_idx = 0
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        start_idx = int(f.read().strip())

indices = list(range(start_idx, length))

pool = Pool(NUM_WORKERS)

dst_txn = dst_env.begin(write=True)
cnt = 0

for result in tqdm(pool.imap_unordered(process_one, indices), total=len(indices)):
    if result is None:
        continue

    idx, data = result
    key = str(idx).encode("ascii")
    dst_txn.put(key, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    cnt += 1

    if cnt % COMMIT_INTERVAL == 0:
        dst_txn.commit()
        with open(CHECKPOINT, "w") as f:
            f.write(str(idx + 1))
        dst_txn = dst_env.begin(write=True)

dst_txn.commit()
with open(CHECKPOINT, "w") as f:
    f.write(str(length))

pool.close()
pool.join()






