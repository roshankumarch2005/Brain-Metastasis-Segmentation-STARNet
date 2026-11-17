#!/bin/bash
# =============================================================================
# launch_local_binary_configs_2_and_3.sh
# -----------------------------------------------------------------------------
# Launches the final 2 dense configurations for Dataset 105 locally in tmux:
#   - Binary Config 2: nnU-Net + Standalone SIAM Attention (31.2M) on GPU 0
#   - Binary Config 3: nnU-Net + Standalone BAA Sensitive 1.5 (31.2M) on GPU 2
# =============================================================================

set -e

BASE_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

mkdir -p "$BASE_DIR/training_logs"

echo "=========================================================="
echo "🚀 Spawning Final Dense Binary Configurations 2 & 3 in tmux"
echo "  • Config 2 (Standalone SIAM) on GPU 0 in tmux 'binary_config2'"
echo "  • Config 3 (Standalone BAA 1.5) on GPU 2 in tmux 'binary_config3'"
echo "=========================================================="

# 1. Config 2 in tmux
if tmux has-session -t binary_config2 2>/dev/null; then
    echo "⚠️  Session 'binary_config2' already exists."
else
    echo "▶️  Spawning Binary Config 2 in tmux 'binary_config2'..."
    tmux new-session -d -s binary_config2 "bash /mnt/DATA/EE23B009/EE23B009_BTP/launch_binary_config2_siam_gpu0.sh"
fi

# 2. Config 3 in tmux
if tmux has-session -t binary_config3 2>/dev/null; then
    echo "⚠️  Session 'binary_config3' already exists."
else
    echo "▶️  Spawning Binary Config 3 in tmux 'binary_config3'..."
    tmux new-session -d -s binary_config3 "bash /mnt/DATA/EE23B009/EE23B009_BTP/launch_binary_config3_baa_gpu2.sh"
fi

echo ""
echo "✅ Both sessions successfully spawned!"
tmux ls
