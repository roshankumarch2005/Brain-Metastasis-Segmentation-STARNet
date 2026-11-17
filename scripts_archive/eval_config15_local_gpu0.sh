#!/bin/bash
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
PRED_DIR="$RESULTS_DIR/predictions_config15_flagship_final"
CSV_OUT="$BASE_DIR/config15_flagship_test30_metrics.csv"
LOG_OUT="$BASE_DIR/config15_eval_output.log"

echo "=================================================================="
echo "🎯 STARTING 30-CASE TEST EVALUATION FOR FLAGSHIP CONFIG 15 ON GPU 0"
echo "=================================================================="

mkdir -p "$PRED_DIR"

echo "🚀 [1/2] Running nnUNetv2_predict on 30 test cases..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -i "$INPUT_IMAGES" \
    -o "$PRED_DIR" \
    -d 105 -c 3d_fullres -f 0 \
    -tr nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA \
    -chk checkpoint_final.pth

echo "📊 [2/2] Calculating test metrics (Dice & HD95 with MedPy)..."
$PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_DIR" "$CSV_OUT" | tee "$LOG_OUT"

echo "✅ Evaluation completed successfully! Results written to $CSV_OUT"
