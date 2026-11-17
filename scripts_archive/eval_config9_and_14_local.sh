#!/bin/bash
# =============================================================================
# eval_config9_and_14_local.sh (Run on Local Workstation using free GPU 3)
# -----------------------------------------------------------------------------
# Runs 30-case inference and evaluation locally for:
#   1. Config 9  (Depthwise + SIAM)
#   2. Config 14 (Depthwise + Method 1 + Curriculum BAA)
# =============================================================================

set -e

export CUDA_VISIBLE_DEVICES=3
export PATH="/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"
export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

BASE_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

PYTHON_BIN="/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3"
INPUT_IMAGES="$nnUNet_raw/Dataset102_Restored/imagesTs"

echo "=================================================================="
echo "🚀 LOCAL INFERENCE & EVALUATION ON GPU 3"
echo "=================================================================="

# -----------------------------------------------------------------------------
# 1. Evaluate Configuration 9 (Depthwise + SIAM)
# -----------------------------------------------------------------------------
PRED_DIR_9="$nnUNet_results/Dataset102_Restored/predictions_config9_depthwise_siam"
CSV_OUT_9="$BASE_DIR/config9_depthwise_siam_test30_metrics.csv"
CHK_9="$nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

if [ -f "$CHK_9" ]; then
    echo -e "\n▶️ [1/2] Running Config 9 (Depthwise + SIAM)..."
    mkdir -p "$PRED_DIR_9"
    nnUNetv2_predict -i "$INPUT_IMAGES" -o "$PRED_DIR_9" -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_SIAM -chk checkpoint_final.pth
    $PYTHON_BIN "$BASE_DIR/evaluate_multiclass_test30_general.py" "$PRED_DIR_9" "$CSV_OUT_9"
    echo "✅ Config 9 evaluated! CSV saved at: $CSV_OUT_9"
else
    echo "⚠️ Checkpoint for Config 9 not found at $CHK_9 yet."
fi

# -----------------------------------------------------------------------------
# 2. Evaluate Configuration 14 (Depthwise + Method 1 + Curriculum BAA)
# -----------------------------------------------------------------------------
PRED_DIR_14="$nnUNet_results/Dataset102_Restored/predictions_config14_depthwise_method1_curriculum_baa"
CSV_OUT_14="$BASE_DIR/config14_depthwise_method1_curriculum_baa_test30_metrics.csv"
CHK_14="$nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

if [ -f "$CHK_14" ]; then
    echo -e "\n▶️ [2/2] Running Config 14 (Depthwise + M1 + Curriculum BAA)..."
    mkdir -p "$PRED_DIR_14"
    nnUNetv2_predict -i "$INPUT_IMAGES" -o "$PRED_DIR_14" -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA -chk checkpoint_final.pth
    $PYTHON_BIN "$BASE_DIR/evaluate_multiclass_test30_general.py" "$PRED_DIR_14" "$CSV_OUT_14"
    echo "✅ Config 14 evaluated! CSV saved at: $CSV_OUT_14"
else
    echo "⚠️ Checkpoint for Config 14 not found at $CHK_14 yet."
fi

echo -e "\n🎉 Local Evaluation Complete!"

