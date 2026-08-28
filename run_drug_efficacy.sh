#!/bin/bash
#SBATCH --job-name=drug_eff_full
#SBATCH --output=/home/student/s230005071/shu/code2/summary_results/10mineeg/logs/drug_eff_full_%j.log
#SBATCH --error=/home/student/s230005071/shu/code2/summary_results/10mineeg/logs/drug_eff_full_%j.err
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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

# 基于三分类 AST-Net_GCN 模型的药物疗效评估 (全量均衡版, 4组×3时相)
# 扫描 cwt_images_all_phases 全部图片, 含 pilo 建模对照
# 模型: 20260804_184146_ASTNet_Standard_3Class/AST-Net_GCN/best_AST-Net_GCN.pth
# 输出: summary_results/10mineeg/results/20260804_184146_ASTNet_Standard_3Class/AST-Net_GCN/drug_efficacy_full/
python drug_efficacy_analysis_animal.py

echo "======================================"
echo "作业结束时间: $(date)"
echo "======================================"
