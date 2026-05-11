#!/bin/bash
# finetune_common.sh - 通用分子属性预测微调脚本


# 从参数传入
pre_train_name=$1  # 预训练模型名称
ckp=$2
task_name=$3
task_num=$4
loss_func=$5
lr=$6
batch_size=$7
epoch=$8
dropout=$9
warmup=${10}


echo "pre_train_name: $pre_train_name"
echo "ckp: $ckp"
echo "task_name: $task_name"
echo "task_num: $task_num"
echo "loss_func: $loss_func"
echo "lr: $lr"
echo "batch_size: $batch_size"
echo "epoch: $epoch"
echo "dropout: $dropout"
echo "warmup: $warmup"



# data_path="/home/aizoo/data/workspace/tanhaojiang/mol_property/unimol_data/molecular_property_prediction_fg"  # 数据路径
data_path="/home/aizoo/data/workspace/tanhaojiang/mol_toxicity/data_lmdb/molecular_toxicity_prediction_fg"  # 数据路径
dict_name="dict.txt"


weight_path="./pretrain_res/$pre_train_name/$ckp.pt"
save_dir="./save_finetune/$pre_train_name/$ckp"_"$task_name"_$(date +%Y%m%d_%H%M%S)



# 可选参数（有默认值）
conf_size=${11:-11}
only_polar=${12:-0}
local_batch_size=${13:-32}
n_gpu=${14:-1}
MASTER_PORT=${15:-10086}
seed=${16:-0}

# 检查必需参数
if [ -z "$task_name" ] || [ -z "$task_num" ] || [ -z "$loss_func" ] || [ -z "$lr" ] || \
   [ -z "$batch_size" ] || [ -z "$epoch" ] || [ -z "$dropout" ] || [ -z "$warmup" ]; then
    echo "错误: 缺少必需参数"
    echo "用法: $0 <task_name> <task_num> <loss_func> <lr> <batch_size> <epoch> <dropout> <warmup> [conf_size] [only_polar] [local_batch_size] [n_gpu] [MASTER_PORT] [seed]"
    echo "示例: $0 bace 2 finetune_cross_entropy 1e-4 64 60 0.1 0.06"
    exit 1
fi



# 设置评估指标
if [ "$task_name" = "qm7" ] || [ "$task_name" = "qm8" ] || [ "$task_name" = "qm9" ]; then
    metric="valid_agg_mae"
elif [ "$task_name" = "esol" ] || [ "$task_name" = "freesolv" ] || [ "$task_name" = "lipo" ]; then
    metric="valid_agg_rmse"
else 
    metric="valid_agg_auc"
fi

# 创建保存目录
mkdir -p $save_dir
cp ./scripts/finetune_common.sh $save_dir
cp ./scripts/run_all_finetune.sh $save_dir
cp -r ./unimol $save_dir/unimol

# 设置环境变量
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

# 计算更新频率
update_freq=$((batch_size / local_batch_size))

# 检查是否多GPU
if [ "$n_gpu" -gt 1 ]; then
    DIST_LAUNCH="python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT"
else
    DIST_LAUNCH=""
fi

# 根据任务类型决定是否加 flag
EXTRA_ARGS=""

if [ "$metric" = "valid_agg_auc" ]; then
    EXTRA_ARGS="--maximize-best-checkpoint-metric"
fi


# 运行训练
$DIST_LAUNCH $(which unicore-train) $data_path \
    --task-name $task_name \
    --keep-last-epochs 2 \
    --user-dir ./unimol \
    --train-subset train \
    --valid-subset valid,test \
    --conf-size $conf_size \
    --num-workers 8 \
    --ddp-backend=c10d \
    --dict-name $dict_name \
    --task mol_finetune \
    --loss $loss_func \
    --arch unimol_base \
    --classification-head-name $task_name \
    --num-classes $task_num \
    --optimizer adam \
    --adam-betas "(0.9, 0.99)" \
    --adam-eps 1e-6 \
    --clip-norm 1.0 \
    --lr-scheduler polynomial_decay \
    --lr $lr \
    --warmup-ratio $warmup \
    --max-epoch $epoch \
    --batch-size $local_batch_size \
    --pooler-dropout $dropout \
    --update-freq $update_freq \
    --seed $seed \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --log-interval 100 \
    --log-format simple \
    --validate-interval 1 \
    --finetune-from-model $weight_path \
    --best-checkpoint-metric $metric \
    --patience 20 \
    --save-dir $save_dir \
    --only-polar $only_polar\
    $EXTRA_ARGS

echo "微调完成! 结果保存在: $save_dir"









