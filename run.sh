#!/bin/bash
#SBATCH --job-name=astnet_multi       # 作业名称
#SBATCH --output=/home/student/s230005071/shu/code2/logs/train_%j.log   # 输出日志（绝对路径）
#SBATCH --error=/home/student/s230005071/shu/code2/logs/train_%j.err    # 错误日志（绝对路径）
#SBATCH --nodes=1                     # 使用1个节点
#SBATCH --time=96:0:00               # 运行时间限制
#SBATCH --partition=gpu               # 指定GPU分区
#SBATCH --gres=gpu:2                  # 请求2个GPU（可修改为4或8）
#SBATCH --cpus-per-task=8             # 8个CPU核心（每个GPU配4个）
#SBATCH --mem=32G                     # 32GB内存

PROJECT_DIR="/home/student/s230005071/shu/code2"
mkdir -p ${PROJECT_DIR}/logs

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

# ==============================================
# 训练模式选择（取消注释对应模式）
# ==============================================

# 模式1: 训练所有模型（默认）
#python main.py --config config/config.yaml

# 模式2: 训练单个模型（替换为你想训练的模型名）
python main.py --config config/config.yaml --model AST-Net_GCN_resnext50_32x4d

# 模式3: 训练选中的多个模型（用逗号分隔）
# python main.py --config config/config.yaml --models AST-Net_GCN_resnext50_32x4d,EEGNet,ST-GCN_EEG

# 可用模型名称列表:
# - AST-Net_GCN_resnext50_32x4d (AST-Net + ResNeXt50 backbone) ← 当前主模型
# - AST-Net_GCN_resnet18        (AST-Net + ResNet18 backbone)
# - Baseline_ResNeXt_resnet18   (Baseline + ResNet18)
# - Baseline_EffNet_efficientnet_b0 (Baseline + EfficientNet B0)
# - Baseline_CNNLSTM_resnet18   (Baseline + CNN-LSTM + ResNet18)
# - EEGNet                      (经典EEG分类模型)
# - ST-GCN_EEG                  (时空图卷积网络)
# - EEGConformer_GCN            (Transformer+GCN融合模型)

echo "======================================"
echo "作业结束时间: $(date)"
echo "======================================"