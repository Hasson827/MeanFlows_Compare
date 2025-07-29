#!/bin/bash
set -e

# 1. 运行训练脚本
source /home/hongshuozhao/.venv/bin/activate
cd MeanFlows_Compare
nohup accelerate launch train.py > train.log 2>&1 &
echo "Training task submitted! Check train.log for output."