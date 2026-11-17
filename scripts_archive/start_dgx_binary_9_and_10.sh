#!/bin/bash
# =============================================================================
# start_dgx_binary_9_and_10.sh (Run this directly on DGX)
# -----------------------------------------------------------------------------
# Launches the 2 remaining Depthwise Ablation Configurations in tmux:
#   - GPU 3: Binary Config 9  (Depthwise + SIAM)
#   - GPU 4: Binary Config 10 (Depthwise + BAA Sensitive 1.5)
#
# (You can also pass custom GPU IDs: ./start_dgx_binary_9_and_10.sh 3 4)
# =============================================================================

set -e

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

mkdir -p training_logs

GPU_CONF9=${1:-3}
GPU_CONF10=${2:-4}

echo "=========================================================="
echo "🚀 Starting Binary Configs 9 & 10 on DGX"
echo "  • Config 9 (Depthwise + SIAM) on GPU $GPU_CONF9"
echo "  • Config 10 (Depthwise + BAA Sensitive) on GPU $GPU_CONF10"
echo "=========================================================="

# 1. Config 9
if tmux has-session -t binary_config9_gpu${GPU_CONF9} 2>/dev/null; then
    echo "⚠️  Session 'binary_config9_gpu${GPU_CONF9}' already exists. Skipping launch."
else
    echo "▶️  Launching Binary Config 9 on GPU $GPU_CONF9 in session 'binary_config9_gpu${GPU_CONF9}'..."
    tmux new-session -d -s binary_config9_gpu${GPU_CONF9} \
        "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=$GPU_CONF9 ./launch_dgx_binary_config9_depthwise_siam.sh"
fi

# 2. Config 10
if tmux has-session -t binary_config10_gpu${GPU_CONF10} 2>/dev/null; then
    echo "⚠️  Session 'binary_config10_gpu${GPU_CONF10}' already exists. Skipping launch."
else
    echo "▶️  Launching Binary Config 10 on GPU $GPU_CONF10 in session 'binary_config10_gpu${GPU_CONF10}'..."
    tmux new-session -d -s binary_config10_gpu${GPU_CONF10} \
        "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=$GPU_CONF10 ./launch_dgx_binary_config10_depthwise_baa.sh"
fi

echo ""
echo "=========================================================="
echo "✅ Both sessions launched successfully on DGX!"
echo "Check active sessions with: tmux ls"
echo "Attach with: tmux a -t binary_config9_gpu${GPU_CONF9}"
echo "=========================================================="
