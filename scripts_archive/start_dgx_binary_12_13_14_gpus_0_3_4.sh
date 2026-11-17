#!/bin/bash
# start_dgx_binary_12_13_14_gpus_0_3_4.sh
# ----------------------------------------------------
# Starts Binary Configurations 12, 13, and 14 in tmux on DGX A100:
# GPU 0: Config 12 (Depthwise + Method 1)
# GPU 3: Config 13 (Depthwise + Method 1 + SIAM)
# GPU 4: Config 14 (Depthwise + Method 1 + Curriculum BAA)

set -e

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

mkdir -p training_logs

echo "=========================================================="
echo "🚀 Starting 3 Binary Configurations on DGX (GPUs 0, 3, 4)"
echo "=========================================================="

# 1. Config 12 on GPU 0
if tmux has-session -t binary_config12_gpu0 2>/dev/null; then
    echo "⚠️  Session 'binary_config12_gpu0' already exists. Skipping launch."
else
    echo "▶️  Launching Config 12 (Depthwise + M1) on GPU 0 in session 'binary_config12_gpu0'..."
    tmux new-session -d -s binary_config12_gpu0 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=0 ./launch_dgx_binary_config12_depthwise_method1.sh"
fi

# 2. Config 13 on GPU 3
if tmux has-session -t binary_config13_gpu3 2>/dev/null; then
    echo "⚠️  Session 'binary_config13_gpu3' already exists. Skipping launch."
else
    echo "▶️  Launching Config 13 (Depthwise + M1 + SIAM) on GPU 3 in session 'binary_config13_gpu3'..."
    tmux new-session -d -s binary_config13_gpu3 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=3 ./launch_dgx_binary_config13_depthwise_method1_siam.sh"
fi

# 3. Config 14 on GPU 4
if tmux has-session -t binary_config14_gpu4 2>/dev/null; then
    echo "⚠️  Session 'binary_config14_gpu4' already exists. Skipping launch."
else
    echo "▶️  Launching Config 14 (Depthwise + M1 + Curriculum BAA) on GPU 4 in session 'binary_config14_gpu4'..."
    tmux new-session -d -s binary_config14_gpu4 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=4 ./launch_dgx_binary_config14_depthwise_method1_curriculum_baa.sh"
fi

echo ""
echo "=========================================================="
echo "✅ All 3 sessions launched successfully in background!"
echo "Check active sessions with: tmux ls"
echo "Attach to session: tmux a -t <session_name>"
echo "=========================================================="

