#!/bin/bash
# =============================================================================
# run_eval_binary_local_configs_2_and_3.sh
# -----------------------------------------------------------------------------
# Evaluates the 2 newly completed dense binary models on the 30 held-out test scans:
#   - Binary Config 2: nnU-Net + Standalone SIAM Attention (31.2M) on GPU 0
#   - Binary Config 3: nnU-Net + Standalone BAA Sensitive 1.5 (31.2M) on GPU 2
# =============================================================================

set -e

BASE_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

export nnUNet_raw="$BASE_DIR/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/nnunet_data/nnUNet_results"
export PATH="$BASE_DIR/brats_env/bin:$PATH"
PYTHON_BIN="$BASE_DIR/brats_env/bin/python3"

INPUT_IMAGES="$nnUNet_raw/Dataset105_UCSF_Saluja_Exact/imagesTs"
RESULTS_DIR="$nnUNet_results/Dataset105_UCSF_Saluja_Exact"

echo "=================================================================="
echo "🎯 STARTING 30-CASE TEST EVALUATION FOR BINARY CONFIGS 2 & 3"
echo "  • Config 2 (Standalone SIAM, 31.2M) on GPU 0"
echo "  • Config 3 (Standalone BAA 1.5, 31.2M) on GPU 2"
echo "=================================================================="

CHK_2="$RESULTS_DIR/nnUNetTrainerSIAM__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
CHK_3="$RESULTS_DIR/nnUNetTrainer_BAA_Sensitive__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

if [ ! -f "$CHK_2" ]; then
    echo "❌ ERROR: Config 2 checkpoint not found at $CHK_2!"
    exit 1
fi

if [ ! -f "$CHK_3" ]; then
    echo "❌ ERROR: Config 3 checkpoint not found at $CHK_3!"
    exit 1
fi

echo "✅ Found Config 2 Checkpoint: $CHK_2"
echo "✅ Found Config 3 Checkpoint: $CHK_3"

run_gpu0_config2() {
    echo -e "\n[GPU 0] 🚀 Starting Config 2 (Standalone SIAM) prediction..."
    PRED_2="$RESULTS_DIR/predictions_config2_dense_siam"
    mkdir -p "$PRED_2"
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_2" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainerSIAM \
        -chk checkpoint_final.pth
    
    echo -e "\n[GPU 0] 📊 Calculating metrics for Config 2..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_2" "$BASE_DIR/config2_dense_siam_test30_metrics.csv" | tee "$BASE_DIR/config2_eval_output.txt"
}

run_gpu2_config3() {
    echo -e "\n[GPU 2] 🚀 Starting Config 3 (Standalone BAA 1.5) prediction..."
    PRED_3="$RESULTS_DIR/predictions_config3_dense_baa"
    mkdir -p "$PRED_3"
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_3" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_BAA_Sensitive \
        -chk checkpoint_final.pth
    
    echo -e "\n[GPU 2] 📊 Calculating metrics for Config 3..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_3" "$BASE_DIR/config3_dense_baa_test30_metrics.csv" | tee "$BASE_DIR/config3_eval_output.txt"
}

echo "▶️ Launching evaluations on GPU 0 and GPU 2..."
run_gpu0_config2 > "$BASE_DIR/eval_config2_gpu0.log" 2>&1 &
PID_GPU0=$!

run_gpu2_config3 > "$BASE_DIR/eval_config3_gpu2.log" 2>&1 &
PID_GPU2=$!

echo "▶️ GPU 0 PID: $PID_GPU0 (Config 2: Standalone SIAM)"
echo "▶️ GPU 2 PID: $PID_GPU2 (Config 3: Standalone BAA 1.5)"
echo "Logs: $BASE_DIR/eval_config2_gpu0.log and $BASE_DIR/eval_config3_gpu2.log"

wait $PID_GPU0
wait $PID_GPU2

echo "🎉 ALL EVALUATIONS COMPLETE FOR CONFIGS 2 & 3!"
