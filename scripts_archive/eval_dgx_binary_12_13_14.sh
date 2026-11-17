#!/bin/bash
# =============================================================================
# eval_dgx_binary_12_13_14.sh
# -----------------------------------------------------------------------------
# Run this directly on DGX to evaluate the finished binary models:
#   - Config 12 (Depthwise + Method 1)
#   - Config 13 (Depthwise + Method 1 + SIAM)
#   - Config 14 (Depthwise + Method 1 + Curriculum BAA)
#
# Generates 3D predictions on the 30 held-out test cases of Dataset105,
# and calculates physical spacing Dice and HD95 metrics.
# =============================================================================

set -e

BASE_DIR="/Data3/EE23B009/EE23B009_BTP"
cd "$BASE_DIR"

export nnUNet_raw="$BASE_DIR/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/nnunet_data/nnUNet_results"
export PATH="/Data3/EE23B009/brats_env/bin:/Data3/EE23B009/EE23B009_BTP/brats_env/bin:$PATH"

PYTHON_BIN="/Data3/EE23B009/brats_env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="/Data3/EE23B009/EE23B009_BTP/brats_env/bin/python3"
fi

INPUT_IMAGES="$nnUNet_raw/Dataset105_UCSF_Saluja_Exact/imagesTs"
RESULTS_DIR="$nnUNet_results/Dataset105_UCSF_Saluja_Exact"

echo "=================================================================="
echo "🎯 EVALUATING BINARY CONFIGS 12, 13, 14 ON DGX (DATASET 105)"
echo "=================================================================="

# -----------------------------------------------------------------------------
# 1. Evaluate Config 12 (Depthwise + Method 1)
# -----------------------------------------------------------------------------
PRED_12="$RESULTS_DIR/predictions_config12_depthwise_method1"
if [ -d "$RESULTS_DIR/nnUNetTrainer_Method3_InterLevelAll__nnUNetPlans__3d_fullres/fold_0" ]; then
    echo -e "\n▶️ [1/3] Predicting Config 12 (Depthwise + Method 1) on GPU 0..."
    mkdir -p "$PRED_12"
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_12" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_InterLevelAll \
        -chk checkpoint_final.pth
    
    echo "📊 Evaluating Config 12 metrics..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_12" | tee "$BASE_DIR/config12_depthwise_method1_eval.txt"
fi

# -----------------------------------------------------------------------------
# 2. Evaluate Config 13 (Depthwise + Method 1 + SIAM)
# -----------------------------------------------------------------------------
PRED_13="$RESULTS_DIR/predictions_config13_depthwise_method1_siam"
if [ -d "$RESULTS_DIR/nnUNetTrainer_Method3_InterLevel_SIAM__nnUNetPlans__3d_fullres/fold_0" ]; then
    echo -e "\n▶️ [2/3] Predicting Config 13 (Depthwise + Method 1 + SIAM) on GPU 3..."
    mkdir -p "$PRED_13"
    CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_13" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_InterLevel_SIAM \
        -chk checkpoint_final.pth
    
    echo "📊 Evaluating Config 13 metrics..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_13" | tee "$BASE_DIR/config13_depthwise_method1_siam_eval.txt"
fi

# -----------------------------------------------------------------------------
# 3. Evaluate Config 14 (Depthwise + Method 1 + Curriculum BAA)
# -----------------------------------------------------------------------------
PRED_14="$RESULTS_DIR/predictions_config14_depthwise_method1_curriculum_baa"
if [ -d "$RESULTS_DIR/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA__nnUNetPlans__3d_fullres/fold_0" ]; then
    echo -e "\n▶️ [3/3] Predicting Config 14 (Depthwise + Method 1 + Curriculum BAA) on GPU 4..."
    mkdir -p "$PRED_14"
    CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict \
        -i "$INPUT_IMAGES" \
        -o "$PRED_14" \
        -d 105 -c 3d_fullres -f 0 \
        -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA \
        -chk checkpoint_final.pth
    
    echo "📊 Evaluating Config 14 metrics..."
    $PYTHON_BIN "$BASE_DIR/evaluate_saluja_test30.py" "$PRED_14" | tee "$BASE_DIR/config14_depthwise_method1_curriculum_baa_eval.txt"
fi

echo -e "\n=================================================================="
echo "✅ All Available Evaluations Completed!"
echo "=================================================================="
