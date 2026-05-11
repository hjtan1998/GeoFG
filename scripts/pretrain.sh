data_path=/home/aizoo/data/workspace/tanhaojiang/mol_property/unimol_data/ligands_fg # replace to your data path
# data_path=/home/aizoo/data/workspace/tanhaojiang/mol_property/H2EMol/data/unimol_format-5m-1k

save_dir=./pretrain_res/pretrained_mol_$(date +%Y%m%d_%H%M%S)/ # replace to your save path

# 创建目录（不存在则创建）
mkdir -p $save_dir
# 复制单个文件
cp ./scripts/pretrain.sh $save_dir
# 复制整个目录（-r）
cp -r ./unimol $save_dir/unimol


n_gpu=1
MASTER_PORT=10086
lr=1e-4
wd=1e-4
batch_size=256
update_freq=1

masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
masked_fg_token_loss=1
masked_fg_coord_loss=5
masked_fg_dist_loss=10
masked_fg_atom_dist_loss=10

# masked_token_loss=0
# masked_coord_loss=0
# masked_dist_loss=0
# masked_fg_token_loss=0
# masked_fg_coord_loss=0
# masked_fg_dist_loss=0
# masked_fg_atom_dist_loss=0

x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
x_fg_norm_loss=0.01
delta_fg_pair_repr_norm_loss=0.01
delta_fg_atom_pair_repr_norm_loss=0.01
# x_fg_norm_loss=0
# delta_fg_pair_repr_norm_loss=0
# delta_fg_atom_pair_repr_norm_loss=0


masked_relation_loss=1


mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=1
warmup_steps=10000
max_steps=90000

CUDA_VISIBLE_DEVICES=0


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
# python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
#        --num-workers 8 --ddp-backend=c10d \
#        --task unimol --loss unimol --arch unimol_base  \
#        --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
#        --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
#        --update-freq $update_freq --seed $seed \
#        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
#        --max-update $max_steps --log-interval 10 --log-format simple \
#        --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
#        --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
#        --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
#        --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
#        --save-dir $save_dir  --only-polar $only_polar


$(which unicore-train) $data_path  --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 16 --ddp-backend=c10d \
       --task unimol --loss unimol --arch unimol_base  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $max_steps \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 10 --log-format simple \
       --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 3 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss --masked-relation-loss $masked_relation_loss \
       --masked-fg-token-loss $masked_fg_token_loss --masked-fg-coord-loss $masked_fg_coord_loss --masked-fg-dist-loss $masked_fg_dist_loss --masked-fg-atom-dist-loss $masked_fg_atom_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --x-fg-norm-loss $x_fg_norm_loss --delta-fg-pair-repr-norm-loss $delta_fg_pair_repr_norm_loss --delta-fg-atom-pair-repr-norm-loss $delta_fg_atom_pair_repr_norm_loss \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size \
       --save-dir $save_dir  --only-polar $only_polar





# sh scripts/pretrain.sh






# /home/aizoo/miniconda3/envs/alidiff/lib/python3.8/site-packages/unicore-0.0.1-py3.8.egg/unicore_cli/train.py


# UniMolLoss




# tensorboard --logdir /data/workspace/tanhaojiang/AI4Sci/Uni-Mol-main/unimol/save_1/tsb/train_inner/

# tensorboard --logdir /data/workspace/tanhaojiang/AI4Sci/Uni-Mol-main/unimol/save_2/tsb


# tensorboard --logdir /data/workspace/tanhaojiang/AI4Sci/Uni-Mol-main/unimol/pretrain_res/pretrained_mol_20251209_113118/tsb

