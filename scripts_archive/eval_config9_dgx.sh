#!/bin/bash
# eval_config9_dgx.sh
# ----------------------------------------------------
# Evaluates Configuration 9 (3D Depthwise + SIAM Attention) on DGX GPU 1
set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

BASE_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

INPUT_IMAGES="$nnUNet_raw/Dataset102_Restored/imagesTs"
PRED_DIR="$nnUNet_results/Dataset102_Restored/predictions_config9_depthwise_siam"
CSV_OUT="$BASE_DIR/config9_depthwise_siam_test30_metrics.csv"

mkdir -p "$PRED_DIR"

echo "=================================================================="
echo "🚀 [1/2] Generating 30-case Predictions on DGX GPU 1..."
echo "=================================================================="
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
    -i "$INPUT_IMAGES" \
    -o "$PRED_DIR" \
    -d 102 -c 3d_fullres -f 0 \
    -tr nnUNetTrainer_Method3_SIAM \
    -chk checkpoint_final.pth

echo "=================================================================="
echo "📊 [2/2] Calculating Exact Metrics on 30 Held-Out Test Scans..."
echo "=================================================================="
PYTHON_BIN="/Data3/EE23B009/brats_env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="/Data3/EE23B009/EE23B009_BTP/brats_env/bin/python3"
fi

$PYTHON_BIN "$BASE_DIR/evaluate_multiclass_test30_general.py" "$PRED_DIR" "$CSV_OUT"

echo "=================================================================="
echo "✅ Config 9 Evaluation Complete! CSV saved at: $CSV_OUT"
echo "=================================================================="
