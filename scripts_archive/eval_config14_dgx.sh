#!/bin/bash
# eval_config14_dgx.sh
# ----------------------------------------------------
# Evaluates Configuration 14 (3D Depthwise + Method 1 + Curriculum BAA) on DGX GPU 7
set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

BASE_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

INPUT_IMAGES="$nnUNet_raw/Dataset102_Restored/imagesTs"
PRED_DIR="$nnUNet_results/Dataset102_Restored/predictions_config14_depthwise_method1_curriculum_baa"
CSV_OUT="$BASE_DIR/config14_depthwise_method1_curriculum_baa_test30_metrics.csv"

mkdir -p "$PRED_DIR"

echo "=================================================================="
echo "🚀 [1/2] Generating 30-case Predictions on DGX GPU 7..."
echo "=================================================================="
CUDA_VISIBLE_DEVICES=7 nnUNetv2_predict \
    -i "$INPUT_IMAGES" \
    -o "$PRED_DIR" \
    -d 102 -c 3d_fullres -f 0 \
    -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA \
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
echo "✅ Config 14 Evaluation Complete! CSV saved at: $CSV_OUT"
echo "=================================================================="
