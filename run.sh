#!/bin/bash
#SBATCH --job-name=astnet_standard
#SBATCH --output=/home/student/s230005071/shu/code2/summary_results/10mineeg/logs/train_%j.log
#SBATCH --error=/home/student/s230005071/shu/code2/summary_results/10mineeg/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --time=96:0:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G



PROJECT_DIR="/home/student/s230005071/shu/code2"
mkdir -p ${PROJECT_DIR}/summary_results/10mineeg/logs

echo "======================================"
echo "作业开始时间: $(date)"
echo "节点: $(hostname)"
echo "作业ID: ${SLURM_JOB_ID}"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "======================================"

source /home/student/s230005071/miniconda3/etc/profile.d/conda.sh
conda activate torch_stable_cu121

cd ${PROJECT_DIR}

# ========== Standard 数据集 (3类, 8通道) ==========
# config: config/config.yaml
# 标签: Baseline(0), Ictal(1), Chronic(2)

# 模式1: 训练所有模型
python main.py --config config/config.yaml

# 模式2: 训练单个模型
#python main.py --config config/config.yaml --model AST-Net_GCN

# 模式3: 训练选中的多个模型
#python main.py --config config/config.yaml --models AST-Net_GCN,EEGNet,ST-GCN_EEG

# 可用模型名称:
#   AST-Net_GCN                    ← AST多尺度 + GCN (ResNeXt50 backbone)
#   Baseline_MultiScaleResNeXt     ← 仅多尺度CNN (ResNeXt50 + MultiScaleBlock, 无GCN)
#   Baseline_ResNeXt_resnext50_32x4d
#   Baseline_ResNet_resnet18
#   Baseline_EffNet_efficientnet_b0
#   Baseline_CNNLSTM_resnet18
#   EEGNet
#   ST-GCN_EEG
#   EEGConformer                   ← Braindecode官方EEGConformer实现

echo "======================================"
echo "作业结束时间: $(date)"
echo "======================================"
