#!/usr/bin/env bash
# run_all_finetune.sh - 运行所有数据集的微调（分类 + 回归）
# 兼容老 bash / HPC 环境

set -u
# sider pcba freesolv lipo qm7 qm8 qm9


# pretrained_mol_20260122_162052          
# pretrained_mol_20260122_170838
# pretrained_mol_20260123_111858
# pretrained_mol_20260123_192222

pre_train_name="pretrained_mol_20260123_192222"
ckp="checkpoint_2_80000"

mkdir -p "./save_finetune/$pre_train_name"

############################################
# 获取数据集参数
############################################
get_dataset_params() {
    dataset=$1
    param=$2

    case "$dataset" in
        biodegradation)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        carcinogenicity)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        dili)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        eye_corrosion)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        hia)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        bbbp)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 4e-4 ;;
                batch_size) echo 128 ;;
                epoch) echo 40 ;;
                dropout) echo 0 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        bace)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        clintox)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo multi_task_BCE ;;
                lr) echo 5e-5 ;;
                batch_size) echo 256 ;;
                epoch) echo 100 ;;
                dropout) echo 0.5 ;;
                warmup) echo 0.1 ;;
            esac
            ;;
        tox21)
            case "$param" in
                task_num) echo 12 ;;
                loss_func) echo multi_task_BCE ;;
                lr) echo 1e-4 ;;
                batch_size) echo 128 ;;
                epoch) echo 80 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        toxcast)
            case "$param" in
                task_num) echo 617 ;;
                loss_func) echo multi_task_BCE ;;
                lr) echo 1e-4 ;;
                batch_size) echo 64 ;;
                epoch) echo 80 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        sider)
            case "$param" in
                task_num) echo 27 ;;
                loss_func) echo multi_task_BCE ;;
                lr) echo 5e-4 ;;
                batch_size) echo 32 ;;
                epoch) echo 80 ;;
                dropout) echo 0 ;;
                warmup) echo 0.1 ;;
            esac
            ;;
        hiv)
            case "$param" in
                task_num) echo 2 ;;
                loss_func) echo finetune_cross_entropy ;;
                lr) echo 5e-5 ;;
                batch_size) echo 256 ;;
                epoch) echo 5 ;;
                dropout) echo 0.2 ;;
                warmup) echo 0.1 ;;
            esac
            ;;
        pcba)
            case "$param" in
                task_num) echo 128 ;;
                loss_func) echo multi_task_BCE ;;
                lr) echo 1e-4 ;;
                batch_size) echo 128 ;;
                epoch) echo 20 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        muv)
            case "$param" in
                task_num) echo 17 ;;
                loss_func) echo multi_task_BCE ;;
                lr) echo 2e-5 ;;
                batch_size) echo 128 ;;
                epoch) echo 40 ;;
                dropout) echo 0 ;;
                warmup) echo 0 ;;
            esac
            ;;
        esol)
            case "$param" in
                task_num) echo 1 ;;
                loss_func) echo finetune_mse ;;
                lr) echo 5e-4 ;;
                batch_size) echo 256 ;;
                epoch) echo 100 ;;
                dropout) echo 0.2 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        freesolv)
            case "$param" in
                task_num) echo 1 ;;
                loss_func) echo finetune_mse ;;
                lr) echo 8e-5 ;;
                batch_size) echo 64 ;;
                epoch) echo 60 ;;
                dropout) echo 0.2 ;;
                warmup) echo 0.1 ;;
            esac
            ;;
        lipo)
            case "$param" in
                task_num) echo 1 ;;
                loss_func) echo finetune_mse ;;
                lr) echo 1e-4 ;;
                batch_size) echo 32 ;;
                epoch) echo 80 ;;
                dropout) echo 0.1 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        qm7)
            case "$param" in
                task_num) echo 1 ;;
                loss_func) echo finetune_smooth_mae ;;
                lr) echo 3e-4 ;;
                batch_size) echo 32 ;;
                epoch) echo 100 ;;
                dropout) echo 0 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        qm8)
            case "$param" in
                task_num) echo 12 ;;
                loss_func) echo finetune_smooth_mae ;;
                lr) echo 1e-4 ;;
                batch_size) echo 32 ;;
                epoch) echo 40 ;;
                dropout) echo 0 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        qm9)
            case "$param" in
                task_num) echo 3 ;;
                loss_func) echo finetune_smooth_mae ;;
                lr) echo 1e-4 ;;
                batch_size) echo 128 ;;
                epoch) echo 40 ;;
                dropout) echo 0 ;;
                warmup) echo 0.06 ;;
            esac
            ;;
        *)
            echo ""
            ;;
    esac
}

############################################
# 选择运行模式
############################################
MODE=3

if [ -t 0 ]; then
    echo "请选择运行模式:"
    echo "1. 只运行分类任务"
    echo "2. 只运行回归任务"
    echo "3. 运行所有任务"
    echo "4. 运行单个任务"
    printf "请输入选择 (1-4): "
    read MODE
fi

case "$MODE" in
    1)
        DATASETS="bbbp bace clintox tox21 toxcast sider hiv pcba muv"
        ;;
    2)
        DATASETS="esol freesolv lipo qm7 qm8 qm9"
        ;;
    4)
        echo "请输入数据集名称:"
        read SINGLE_DATASET
        DATASETS="$SINGLE_DATASET"
        ;;
    *)
        DATASETS="bbbp bace clintox tox21 toxcast sider hiv pcba muv esol freesolv lipo qm7 qm8 qm9"
        ;;
esac

############################################
# 日志
############################################
LOG_FILE="./save_finetune/$pre_train_name/finetune_all_$(date +%Y%m%d_%H%M%S).log"
echo "开始微调: $(date)" | tee "$LOG_FILE"
echo "数据集: $DATASETS" | tee -a "$LOG_FILE"

############################################
# 主循环
############################################
for DATASET in $DATASETS; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "开始数据集: $DATASET" | tee -a "$LOG_FILE"

    TASK_NUM=$(get_dataset_params "$DATASET" task_num)
    [ -z "$TASK_NUM" ] && echo "未知数据集 $DATASET，跳过" | tee -a "$LOG_FILE" && continue

    LOSS_FUNC=$(get_dataset_params "$DATASET" loss_func)
    LR=$(get_dataset_params "$DATASET" lr)
    BATCH_SIZE=$(get_dataset_params "$DATASET" batch_size)
    EPOCH=$(get_dataset_params "$DATASET" epoch)
    DROPOUT=$(get_dataset_params "$DATASET" dropout)
    WARMUP=$(get_dataset_params "$DATASET" warmup)


    bash ./scripts/finetune_common.sh \
        "$pre_train_name" "$ckp" \
        "$DATASET" "$TASK_NUM" "$LOSS_FUNC" "$LR" "$BATCH_SIZE" \
        "$EPOCH" "$DROPOUT" "$WARMUP" \
        1 0 32 1 10086 4 \
        >> "$LOG_FILE" 2>&1




    if [ $? -ne 0 ]; then
        echo "❌ $DATASET 失败" | tee -a "$LOG_FILE"
    else
        echo "✅ $DATASET 完成" | tee -a "$LOG_FILE"
    fi

    sleep 5
done

echo "全部任务完成: $(date)" | tee -a "$LOG_FILE"



# bash ./scripts/run_all_finetune.sh


# 数据规模较小
# sider bbbp bace clintox tox21 esol freesolv lipo qm7
# 数据规模较大
# hiv pcba muv toxcast qm8 qm9

# bbbp bace clintox tox21 toxcast sider esol freesolv lipo qm7 hiv pcba muv qm8 qm9

# biodegradation carcinogenicity dili eye_corrosion hia
# screen -r 738900



# bbbp        5450M                 30 epochs -285.3 seconds
# bace        5236M                 46 epochs -387.3 seconds
# clintox     12120M                43 epochs -397.8 seconds
# tox21       10806M                37 epochs -1051.7 seconds
# freesolv    2484M                 52 epochs -312.3 seconds
# lipo        9672M                 80 epochs -1386.5 seconds
# qm7         2260M                 48 epochs -1135.6 seconds
# sider       38098M                41 epochs -641.0 seconds
# esol        4118M                 81 epochs -562.9 seconds










