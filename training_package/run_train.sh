#!/bin/bash

# 1. 安全检查：不要误杀其他任务
echo "Checking for old processes..."
# 仅清理包含 'train.py' 且属于当前用户的进程，避免误伤 'ssx' 环境的任务
# pkill -u qxli -f "train.py" 

# 2. 设置一个新的随机端口，防止冲突
PORT=29667

# 3. 后台启动训练
# --nproc_per_node=8: 使用8张卡
# --master_port=$PORT: 指定端口
# nohup ... & : 后台运行，关闭终端不停止
echo "Starting training on port $PORT in background..."
echo "Configuration: BATCH_SIZE=4, ZeRO-3, BF16, Low CPU Mem Usage"

nohup torchrun --nproc_per_node=8 --master_port=$PORT train.py > train.log 2>&1 &

echo "----------------------------------------------------------------"
echo "✅ 训练已在后台启动！(防泄漏版)"
echo "📉 优化措施: 启用自动GC清理 + 降低Worker数量 + 定期清理CUDA缓存"
echo "📝 日志文件: train.log"
echo "👀 查看日志请运行: tail -f train.log"
echo "----------------------------------------------------------------"
