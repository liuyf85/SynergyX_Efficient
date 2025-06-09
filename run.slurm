#!/bin/bash
#
#SBATCH -J my_gpu_job             # 作业名
#SBATCH -N 1                      # 节点数
#SBATCH -n 1                      # 任务数（并行进程数）
#SBATCH -p critical               # 分区名
#SBATCH -A tukw-critical          # 账户名
#SBATCH --gres=gpu:1              # GPU 资源
#SBATCH --mem=32G                 # 总内存
#SBATCH -t 96:00:00               # 最长运行时间 2 小时（可选）
#SBATCH -o my_gpu_job.%j.out      # 标准输出文件
#SBATCH -e my_gpu_job.%j.err      # 标准错误文件


source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate my_pytorch_env
# 运行你的程序，比如：
srun sh run.sh
