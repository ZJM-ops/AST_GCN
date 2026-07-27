#!/bin/bash
#SBATCH --job-name=AST-Net-Parallel
#SBATCH --output=/home/student/s230005071/shu/code2/logs/train_%j.log
#SBATCH --error=/home/student/s230005071/shu/code2/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source /home/student/s230005071/miniconda3/etc/profile.d/conda.sh
conda activate torch_stable_cu121

cd /home/student/s230005071/shu/code2
mkdir -p logs
python main.py
