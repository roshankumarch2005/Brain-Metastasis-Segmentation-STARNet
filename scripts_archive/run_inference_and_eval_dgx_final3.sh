#!/bin/bash
# run_inference_and_eval_dgx_final3.sh
# -----------------------------------------------------------------------------
# Fast inference and 30-case evaluation on DGX for the final 3 models:
#   1. Config 9:  Depthwise + SIAM (nnUNetTrainer_Method3_SIAM)
#   2. Config 10: Depthwise + BAA Sensitive 1.5 (nnUNetTrainer_Method3_BAA_Sensitive)
#   3. Config 14: Depthwise + Method 1 + Curriculum BAA (nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA)
#
# Runs on idle DGX GPUs (GPUs 3, 4, 5 in parallel). Total runtime: ~2 minutes on A100.
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
echo "🚀 DGX FAST INFERENCE & BENCHMARK EVALUATION (FINAL 3 MODELS)"
echo "=================================================================="

PRED_DIR_9="$nnUNet_results/Dataset102_Restored/predictions_config9_depthwise_siam"
PRED_DIR_10="$nnUNet_results/Dataset102_Restored/predictions_config10_depthwise_baa"
PRED_DIR_14="$nnUNet_results/Dataset102_Restored/predictions_config14_depthwise_method1_curriculum_baa"

mkdir -p "$PRED_DIR_9" "$PRED_DIR_10" "$PRED_DIR_14"

# -------------------------------------------------------------------------
# 1. Parallel Inference on GPUs 3, 4, 5
# -------------------------------------------------------------------------
echo ""
echo "▶️ [1/3] Launching Config 9 (Depthwise + SIAM) on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict \
    -i "$INPUT_IMAGES" \
    -o "$PRED_DIR_9" \
    -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_SIAM -chk checkpoint_final.pth &
PID_9=$!

echo "▶️ [2/3] Launching Config 10 (Depthwise + BAA Sensitive) on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nnUNetv2_predict \
    -i "$INPUT_IMAGES" \
    -o "$PRED_DIR_10" \
    -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_BAA_Sensitive -chk checkpoint_final.pth &
PID_10=$!

echo "▶️ [3/3] Launching Config 14 (Depthwise + M1 + Curriculum) on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict \
    -i "$INPUT_IMAGES" \
    -o "$PRED_DIR_14" \
    -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA -chk checkpoint_final.pth &
PID_14=$!

echo "⏳ Waiting for all 3 inference processes to complete..."
wait $PID_9
echo "✅ Config 9 inference complete!"
wait $PID_10
echo "✅ Config 10 inference complete!"
wait $PID_14
echo "✅ Config 14 inference complete!"

# -------------------------------------------------------------------------
# 2. Benchmark Evaluation on 30 Unseen Test Cases
# -------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "📊 COMPUTING BENCHMARK METRICS ON 30 UNSEEN TEST CASES"
echo "=================================================================="

PYTHON_BIN="/Data3/EE23B009/brats_env/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="/Data3/EE23B009/EE23B009_BTP/brats_env/bin/python3"
fi

echo "🔍 [1/3] Evaluating Config 9..."
$PYTHON_BIN "$BASE_DIR/evaluate_multiclass_test30_general.py" \
    "$PRED_DIR_9" "$BASE_DIR/config9_depthwise_siam_test30_metrics.csv"

echo "🔍 [2/3] Evaluating Config 10..."
$PYTHON_BIN "$BASE_DIR/evaluate_multiclass_test30_general.py" \
    "$PRED_DIR_10" "$BASE_DIR/config10_depthwise_baa_test30_metrics.csv"

echo "🔍 [3/3] Evaluating Config 14..."
$PYTHON_BIN "$BASE_DIR/evaluate_multiclass_test30_general.py" \
    "$PRED_DIR_14" "$BASE_DIR/config14_depthwise_method1_curriculum_baa_test30_metrics.csv"

echo ""
echo "=================================================================="
echo "🎉 ALL 3 FINAL MODELS EVALUATED SUCCESSFULLY ON DGX!"
echo "Generated CSV Reports:"
echo "  1. $BASE_DIR/config9_depthwise_siam_test30_metrics.csv"
echo "  2. $BASE_DIR/config10_depthwise_baa_test30_metrics.csv"
echo "  3. $BASE_DIR/config14_depthwise_method1_curriculum_baa_test30_metrics.csv"
echo "=================================================================="
