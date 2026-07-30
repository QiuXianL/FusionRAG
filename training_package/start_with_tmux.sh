#!/bin/bash

# Define session name
SESSION_NAME="rag_training"
PORT=29668

echo "----------------------------------------------------------------"
echo "🚀 准备在 Tmux 会话 '$SESSION_NAME' 中启动训练..."
echo "----------------------------------------------------------------"

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Create new session detached
    tmux new-session -d -s $SESSION_NAME
    
    # Send commands to the session
    # 1. Activate environment (assuming conda env 'py310' based on logs)
    # Adjust this line if your environment activation is different!
    # We use 'source' or direct path if possible. 
    # Based on logs: /home/qxli/miniconda3/envs/py310/bin/python3.10
    # So we probably just need to be in the right directory and use the right python or activate.
    # Let's try to source bashrc or activate conda if possible, but safer to use explicit paths if known.
    # User's shell prompt shows (py310), so we assume 'conda activate py310' works if bashrc is sourced.
    
    tmux send-keys -t $SESSION_NAME "source ~/.bashrc" C-m
    tmux send-keys -t $SESSION_NAME "conda activate py310" C-m
    tmux send-keys -t $SESSION_NAME "cd 'e:\program\RAG - 当前使用\training_package'" C-m # Windows path might fail in Linux shell?
    # Wait, the user is on Linux: (py310) qxli@admin:~/training_package$
    # The 'e:\program...' path in the prompt context <env> is the LOCAL IDE path.
    # The REMOTE path is ~/training_package.
    tmux send-keys -t $SESSION_NAME "cd ~/training_package" C-m
    
    # Run the training command
    # Use a new port to avoid conflicts
    tmux send-keys -t $SESSION_NAME "torchrun --nproc_per_node=8 --master_port=$PORT train.py > train_tmux.log 2>&1" C-m
    
    echo "✅ Tmux 会话已创建并启动训练！"
else
    echo "⚠️  会话 '$SESSION_NAME' 已存在。请先检查或关闭它。"
    echo "   查看: tmux attach -t $SESSION_NAME"
    echo "   关闭: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "管理指南 (Management Guide):"
echo "1. 👀 查看训练进度 (Attach):"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "2. 🚪 退出查看但不停止训练 (Detach):"
echo "   按下 [Ctrl+B]，松开后按 [D]"
echo ""
echo "3. 🛑 停止训练:"
echo "   进入会话后按 [Ctrl+C]"
echo "   或者在外部运行: tmux kill-session -t $SESSION_NAME"
echo ""
echo "4. 📝 查看日志:"
echo "   tail -f train_tmux.log"
echo "----------------------------------------------------------------"
