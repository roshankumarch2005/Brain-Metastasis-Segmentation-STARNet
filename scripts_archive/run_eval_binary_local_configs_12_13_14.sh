#!/bin/bash
# =============================================================================
# run_eval_binary_local_configs_12_13_14.sh
# -----------------------------------------------------------------------------
# Runs full test evaluation on the 30 held-out test scans of Dataset105
# for the 3 finished binary models:
#   - Config 12 (Depthwise + Method 1) on GPU 1
#   - Config 13 (Depthwise + Method 1 + SIAM) on GPU 3
#   - Config 14 (Depthwise + Method 1 + Curriculum BAA) on GPU 1 (sequential)
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
echo "🎯 STARTING 30-CASE TEST EVALUATION FOR BINARY CONFIGS 12, 13, 14"
echo "=================================================================="

# Define worker function for Config 12 & 14 on GPU 1
run_gpu1_jobs() {
    echo -e "\n[GPU 1] 🚀 Starting Config 12 prediction..."
    PRED_12="$RESULTS_DIR/predictions_config12_depthwise_method1"
    mkdir -p "$PRED_12"
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_12" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_InterLevelAll \
        -chk checkpoint_final.pth
    
    echo -e "\n[GPU 1] 📊 Calculating metrics for Config 12..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_12" "$BASE_DIR/config12_depthwise_method1_test30_metrics.csv" | tee "$BASE_DIR/config12_eval_output.txt"
    
    echo -e "\n[GPU 1] 🚀 Starting Config 14 prediction..."
    PRED_14="$RESULTS_DIR/predictions_config14_depthwise_method1_curriculum_baa"
    mkdir -p "$PRED_14"
    CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_14" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA \
        -chk checkpoint_final.pth
    
    echo -e "\n[GPU 1] 📊 Calculating metrics for Config 14..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_14" "$BASE_DIR/config14_depthwise_method1_curriculum_baa_test30_metrics.csv" | tee "$BASE_DIR/config14_eval_output.txt"
}

# Define worker function for Config 13 on GPU 3
run_gpu3_job() {
    echo -e "\n[GPU 3] 🚀 Starting Config 13 prediction..."
    PRED_13="$RESULTS_DIR/predictions_config13_depthwise_method1_siam"
    mkdir -p "$PRED_13"
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_13" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_InterLevel_SIAM \
        -chk checkpoint_final.pth
    
    echo -e "\n[GPU 3] 📊 Calculating metrics for Config 13..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_13" "$BASE_DIR/config13_depthwise_method1_siam_test30_metrics.csv" | tee "$BASE_DIR/config13_eval_output.txt"
}

# Launch in parallel on GPU 1 and GPU 3
run_gpu1_jobs > "$BASE_DIR/eval_binary_gpu1.log" 2>&1 &
PID_GPU1=$!

run_gpu3_job > "$BASE_DIR/eval_binary_gpu3.log" 2>&1 &
PID_GPU3=$!

echo "▶️ GPU 1 process PID: $PID_GPU1 (Config 12 followed by Config 14)"
echo "▶️ GPU 3 process PID: $PID_GPU3 (Config 13)"
echo "Logs are streaming to:"
echo "  - $BASE_DIR/eval_binary_gpu1.log"
echo "  - $BASE_DIR/eval_binary_gpu3.log"

# Wait for both workers to finish
wait $PID_GPU1
wait $PID_GPU3

echo -e "\n=================================================================="
echo "🎉 ALL EVALUATIONS COMPLETE! SUMMARY METRICS:"
echo "=================================================================="
echo "--- Config 12 (Depthwise + Method 1) ---"
tail -n 14 "$BASE_DIR/config12_eval_output.txt"
echo -e "\n--- Config 13 (Depthwise + Method 1 + SIAM) ---"
tail -n 14 "$BASE_DIR/config13_eval_output.txt"
echo -e "\n--- Config 14 (Depthwise + Method 1 + Curriculum BAA) ---"
tail -n 14 "$BASE_DIR/config14_eval_output.txt"
echo "=================================================================="
