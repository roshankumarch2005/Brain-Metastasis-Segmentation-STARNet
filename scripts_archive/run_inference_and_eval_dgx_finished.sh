#!/bin/bash
# run_inference_and_eval_dgx_finished.sh
# -----------------------------------------------------------------------------
# Runs fast inference and 30-case evaluation on DGX for:
#   1. Config 13: Depthwise + Method 1 + SIAM (nnUNetTrainer_Method3_InterLevel_SIAM)
#   2. Config 3: Base nnU-Net + BAA Sensitive (nnUNetTrainer_BAA_Sensitive)
#
# Uses idle DGX GPUs (GPU 3 & GPU 4). Takes ~2-3 minutes per model on A100.
# -----------------------------------------------------------------------------

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

INPUT_IMAGES="$nnUNet_raw/Dataset102_Restored/imagesTs"
BASE_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

echo "=================================================================="
echo "🚀 DGX FAST INFERENCE & BENCHMARK EVALUATION (30 UNSEEN SCANS)"
echo "=================================================================="

# -------------------------------------------------------------------------
# 1. Evaluate Configuration 13 (Depthwise + Method 1 + SIAM)
# -------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🔍 [1/2] Evaluating Config 13: Depthwise + Method 1 + SIAM..."
echo "=================================================================="
PRED_DIR_13="$nnUNet_results/Dataset102_Restored/predictions_config13_depthwise_method1_siam"
mkdir -p "$PRED_DIR_13"

NUM_PREDS_13=$(ls "$PRED_DIR_13"/*.nii.gz 2>/dev/null | wc -l || echo 0)
if [ "$NUM_PREDS_13" -ge 30 ]; then
    echo "✅ Predictions already completed ($NUM_PREDS_13 files found in $PRED_DIR_13). Skipping prediction step!"
else
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_DIR_13" \
        -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM -chk checkpoint_final.pth
fi

/Data3/EE23B009/brats_env/bin/python3 "$BASE_DIR/evaluate_multiclass_test30_general.py" \
    "$PRED_DIR_13" "$BASE_DIR/config13_depthwise_method1_siam_test30_metrics.csv"

# -------------------------------------------------------------------------
# 2. Evaluate Configuration 3 (Base nnU-Net + BAA Sensitive 1.5)
# -------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🔍 [2/2] Evaluating Config 3: Base nnU-Net + BAA Sensitive 1.5..."
echo "=================================================================="
PRED_DIR_3="$nnUNet_results/Dataset102_Restored/predictions_config3_base_baa"
mkdir -p "$PRED_DIR_3"

NUM_PREDS_3=$(ls "$PRED_DIR_3"/*.nii.gz 2>/dev/null | wc -l || echo 0)
if [ "$NUM_PREDS_3" -ge 30 ]; then
    echo "✅ Predictions already completed ($NUM_PREDS_3 files found in $PRED_DIR_3). Skipping prediction step!"
else
    CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_DIR_3" \
        -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_BAA_Sensitive -chk checkpoint_final.pth
fi

/Data3/EE23B009/brats_env/bin/python3 "$BASE_DIR/evaluate_multiclass_test30_general.py" \
    "$PRED_DIR_3" "$BASE_DIR/config3_base_baa_test30_metrics.csv"

echo ""
echo "=================================================================="
echo "🎉 BOTH CONFIG 13 AND CONFIG 3 EVALUATED SUCCESSFULLY!"
echo "Metrics CSVs created:"
echo "  1. $BASE_DIR/config13_depthwise_method1_siam_test30_metrics.csv"
echo "  2. $BASE_DIR/config3_base_baa_test30_metrics.csv"
echo "=================================================================="
