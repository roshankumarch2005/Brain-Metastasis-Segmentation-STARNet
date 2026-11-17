#!/bin/bash
# kill_dgx_gpus_2_7.sh
# ----------------------------------------------------
# Kills all tmux sessions and frees GPU 2 and GPU 7 on DGX A100.

echo "=========================================================="
echo "🛑 Releasing DGX GPU 2 and GPU 7..."
echo "=========================================================="

# 1. Kill specific tmux sessions associated with GPU 2 and GPU 7
SESSIONS=(
    "binary_config11_gpu2"
    "binary_config15_gpu7"
    "binary_config9_gpu2"
    "binary_config10_gpu7"
    "dgx_config10_gpu2"
    "dgx_config14_gpu7"
)

for SESS in "${SESSIONS[@]}"; do
    if tmux has-session -t "$SESS" 2>/dev/null; then
        echo "Killing tmux session: $SESS"
        tmux kill-session -t "$SESS" 2>/dev/null || true
    fi
done

# 2. Check for any remaining tmux sessions with gpu2 or gpu7 in the name
for SESS in $(tmux ls -F '#{session_name}' 2>/dev/null | grep -E "gpu[27]"); do
    echo "Killing matched tmux session: $SESS"
    tmux kill-session -t "$SESS" 2>/dev/null || true
done

# 3. Kill any lingering python / nnUNet processes on GPU 2 and GPU 7
echo "Ensuring all processes on GPU 2 and GPU 7 are terminated..."
fuser -v /dev/nvidia2 2>/dev/null | xargs -r kill -9 2>/dev/null || true
fuser -v /dev/nvidia7 2>/dev/null | xargs -r kill -9 2>/dev/null || true

echo ""
echo "=========================================================="
echo "✅ GPU 2 and GPU 7 released completely!"
echo "Current active tmux sessions:"
tmux ls 2>/dev/null || echo "No active tmux sessions."
echo "=========================================================="

