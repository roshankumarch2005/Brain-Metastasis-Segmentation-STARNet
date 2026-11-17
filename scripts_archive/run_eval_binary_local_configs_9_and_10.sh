#!/bin/bash
# =============================================================================
# run_eval_binary_local_configs_9_and_10.sh
# -----------------------------------------------------------------------------
# Runs full test evaluation on the 30 held-out test scans of Dataset105
# for the 2 completed binary models from DGX:
#   - Config 9  (Depthwise + SIAM) on GPU 1
#   - Config 10 (Depthwise + BAA Sensitive 1.5) on GPU 3
#
# Generates 3D segmentations and saves exact per-case & summary CSVs.
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
echo "🎯 STARTING 30-CASE TEST EVALUATION FOR BINARY CONFIGS 9 & 10"
echo "  • Config 9  (Depthwise + SIAM) on GPU 1"
echo "  • Config 10 (Depthwise + BAA Sensitive) on GPU 3"
echo "=================================================================="

# Check if checkpoints exist
CHK_9="$RESULTS_DIR/nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
if [ ! -f "$CHK_9" ]; then
    CHK_9="$RESULTS_DIR/nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
fi

CHK_10="$RESULTS_DIR/nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
if [ ! -f "$CHK_10" ]; then
    CHK_10="$RESULTS_DIR/nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
fi

if [ ! -f "$CHK_9" ]; then
    echo "❌ ERROR: Config 9 checkpoint not found at $CHK_9!"
    echo "Please ensure weights have been pulled from DGX."
    exit 1
fi

if [ ! -f "$CHK_10" ]; then
    echo "❌ ERROR: Config 10 checkpoint not found at $CHK_10!"
    echo "Please ensure weights have been pulled from DGX."
    exit 1
fi

echo "✅ Found Config 9 Checkpoint: $CHK_9"
echo "✅ Found Config 10 Checkpoint: $CHK_10"

# Define worker function for Config 9 on GPU 1
run_gpu1_config9() {
    echo -e "\n[GPU 1] 🚀 Starting Config 9 (Depthwise + SIAM) prediction..."
    PRED_9="$RESULTS_DIR/predictions_config9_depthwise_siam"
    mkdir -p "$PRED_9"
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_9" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_SIAM \
        -chk "$(basename "$CHK_9")"
    
    echo -e "\n[GPU 1] 📊 Calculating metrics for Config 9..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_9" "$BASE_DIR/config9_depthwise_siam_test30_metrics.csv" | tee "$BASE_DIR/config9_eval_output.txt"
}

# Define worker function for Config 10 on GPU 3
run_gpu3_config10() {
    echo -e "\n[GPU 3] 🚀 Starting Config 10 (Depthwise + BAA Sensitive) prediction..."
    PRED_10="$RESULTS_DIR/predictions_config10_depthwise_baa"
    mkdir -p "$PRED_10"
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_10" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_BAA_Sensitive \
        -chk "$(basename "$CHK_10")"
    
    echo -e "\n[GPU 3] 📊 Calculating metrics for Config 10..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_10" "$BASE_DIR/config10_depthwise_baa_test30_metrics.csv" | tee "$BASE_DIR/config10_eval_output.txt"
}

# Launch in parallel on GPU 1 and GPU 3
run_gpu1_config9 > "$BASE_DIR/eval_config9_gpu1.log" 2>&1 &
PID_GPU1=$!

run_gpu3_config10 > "$BASE_DIR/eval_config10_gpu3.log" 2>&1 &
PID_GPU3=$!

echo "▶️ GPU 1 process PID: $PID_GPU1 (Config 9)"
echo "▶️ GPU 3 process PID: $PID_GPU3 (Config 10)"
echo "Logs are streaming to:"
echo "  - $BASE_DIR/eval_config9_gpu1.log"
echo "  - $BASE_DIR/eval_config10_gpu3.log"

wait $PID_GPU1
wait $PID_GPU3

echo -e "\n=================================================================="
echo "🎉 ALL EVALUATIONS COMPLETE! SUMMARY METRICS:"
echo "=================================================================="
echo "--- Config 9 (Depthwise + SIAM) ---"
tail -n 14 "$BASE_DIR/config9_eval_output.txt"
echo -e "\n--- Config 10 (Depthwise + BAA Sensitive) ---"
tail -n 14 "$BASE_DIR/config10_eval_output.txt"
echo "=================================================================="
