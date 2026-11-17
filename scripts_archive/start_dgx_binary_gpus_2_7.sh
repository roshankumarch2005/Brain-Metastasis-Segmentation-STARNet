#!/bin/bash
# start_dgx_binary_gpus_2_7.sh
# ----------------------------------------------------
# Starts 2 Binary Configurations in tmux on DGX A100 (GPUs 2 & 7):
#
# By default:
#   GPU 2: Binary Config 11 (Depthwise + SIAM + Curriculum BAA)
#   GPU 7: Binary Config 15 (Edge Flagship: Depthwise + M1 + SIAM + Curriculum BAA)
#
# If argument "ablations" is provided (./start_dgx_binary_gpus_2_7.sh ablations):
#   GPU 2: Binary Config 9  (Depthwise + SIAM)
#   GPU 7: Binary Config 10 (Depthwise + BAA Sensitive)

set -e

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$DGX_DIR"

mkdir -p training_logs

MODE=${1:-"flagship"}

if [ "$MODE" == "ablations" ]; then
    echo "=========================================================="
    echo "🚀 Starting Binary Ablations on DGX (GPUs 2 & 7)"
    echo "GPU 2: Binary Config 9  (Depthwise + SIAM)"
    echo "GPU 7: Binary Config 10 (Depthwise + BAA Sensitive)"
    echo "=========================================================="

    # GPU 2: Config 9
    if tmux has-session -t binary_config9_gpu2 2>/dev/null; then
        echo "⚠️  Session 'binary_config9_gpu2' already exists. Skipping launch."
    else
        echo "▶️  Launching Binary Config 9 on GPU 2 in session 'binary_config9_gpu2'..."
        tmux new-session -d -s binary_config9_gpu2 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=2 ./launch_dgx_binary_config9_depthwise_siam.sh"
    fi

    # GPU 7: Config 10
    if tmux has-session -t binary_config10_gpu7 2>/dev/null; then
        echo "⚠️  Session 'binary_config10_gpu7' already exists. Skipping launch."
    else
        echo "▶️  Launching Binary Config 10 on GPU 7 in session 'binary_config10_gpu7'..."
        tmux new-session -d -s binary_config10_gpu7 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=7 ./launch_dgx_binary_config10_depthwise_baa.sh"
    fi

else
    echo "=========================================================="
    echo "🚀 Starting Binary Flagships on DGX (GPUs 2 & 7)"
    echo "GPU 2: Binary Config 11 (Depthwise + SIAM + Curriculum BAA)"
    echo "GPU 7: Binary Config 15 (Depthwise + M1 + SIAM + Curriculum BAA Flagship)"
    echo "=========================================================="

    # GPU 2: Config 11
    if tmux has-session -t binary_config11_gpu2 2>/dev/null; then
        echo "⚠️  Session 'binary_config11_gpu2' already exists. Skipping launch."
    else
        echo "▶️  Launching Binary Config 11 on GPU 2 in session 'binary_config11_gpu2'..."
        tmux new-session -d -s binary_config11_gpu2 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=2 ./launch_dgx_binary_config11_depthwise_siam_curriculum_baa.sh"
    fi

    # GPU 7: Config 15
    if tmux has-session -t binary_config15_gpu7 2>/dev/null; then
        echo "⚠️  Session 'binary_config15_gpu7' already exists. Skipping launch."
    else
        echo "▶️  Launching Binary Config 15 on GPU 7 in session 'binary_config15_gpu7'..."
        tmux new-session -d -s binary_config15_gpu7 "cd $DGX_DIR && CUDA_VISIBLE_DEVICES=7 ./launch_dgx_binary_config15_depthwise_method1_siam_curriculum_baa.sh"
    fi
fi

echo ""
echo "=========================================================="
echo "✅ Both sessions launched successfully on DGX in background!"
echo "Check active sessions with: tmux ls"
echo "Attach to GPU 2 session:    tmux a -t binary_config11_gpu2  (or binary_config9_gpu2)"
echo "Attach to GPU 7 session:    tmux a -t binary_config15_gpu7  (or binary_config10_gpu7)"
echo "=========================================================="

