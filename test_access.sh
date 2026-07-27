#!/bin/bash
#SBATCH --job-name=test_access
#SBATCH --output=/home/student/s230005071/shu/code2/logs/test_access_%j.log
#SBATCH --error=/home/student/s230005071/shu/code2/logs/test_access_%j.err
#SBATCH --nodes=1
#SBATCH --time=1:0:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

echo "======================================"
echo "节点: $(hostname)"
echo "当前目录: $(pwd)"
echo "用户: $(whoami)"
echo "======================================"

echo "检查 s230005041 目录访问:"
ls -la /home/student/s230005041/chb-mit/output/shu_format_split/ 2>&1 | head -20

echo "======================================"
echo "测试 Python 读取文件:"
python3 -c "
import os
path = '/home/student/s230005041/chb-mit/output/shu_format_split/train_images.txt'
if os.path.exists(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    print(f'成功读取 {len(lines)} 行')
else:
    print(f'文件不存在: {path}')
"
echo "======================================"