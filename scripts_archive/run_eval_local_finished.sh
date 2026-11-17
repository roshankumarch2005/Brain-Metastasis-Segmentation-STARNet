#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

INPUT_IMAGES="$nnUNet_raw/Dataset102_Restored/imagesTs"
BASE_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
PYTHON="$BASE_DIR/brats_env/bin/python"
PREDICT="$BASE_DIR/brats_env/bin/nnUNetv2_predict"

cd "$BASE_DIR"

PRED_DIR_12="$nnUNet_results/Dataset102_Restored/predictions_config12_depthwise_method1"
PRED_DIR_15="$nnUNet_results/Dataset102_Restored/predictions_config15_depthwise_method1_siam_curriculum"
mkdir -p "$PRED_DIR_12" "$PRED_DIR_15"

echo "=================================================================="
echo "🚀 RUNNING PARALLEL INFERENCE ON LOCAL WORKSTATION (GPUS 0 & 1)"
echo "=================================================================="

# Check if predictions already done for Config 12
NUM_12=$(ls "$PRED_DIR_12"/*.nii.gz 2>/dev/null | wc -l || echo 0)
if [ "$NUM_12" -ge 30 ]; then
    echo "✅ Config 12 predictions already exist ($NUM_12 files)."
else
    echo "▶️ Launching Config 12 Inference on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 $PREDICT \
        -i "$INPUT_IMAGES" \
        -o "$PRED_DIR_12" \
        -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_InterLevelAll -chk checkpoint_final.pth &
    PID_12=$!
fi

# Check if predictions already done for Config 15
NUM_15=$(ls "$PRED_DIR_15"/*.nii.gz 2>/dev/null | wc -l || echo 0)
if [ "$NUM_15" -ge 30 ]; then
    echo "✅ Config 15 predictions already exist ($NUM_15 files)."
else
    echo "▶️ Launching Config 15 Inference on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 $PREDICT \
        -i "$INPUT_IMAGES" \
        -o "$PRED_DIR_15" \
        -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA -chk checkpoint_final.pth &
    PID_15=$!
fi

# Wait for both inference processes
if [ -n "$PID_12" ]; then
    echo "⏳ Waiting for Config 12 (PID $PID_12) to finish..."
    wait $PID_12
    echo "✅ Config 12 inference complete!"
fi

if [ -n "$PID_15" ]; then
    echo "⏳ Waiting for Config 15 (PID $PID_15) to finish..."
    wait $PID_15
    echo "✅ Config 15 inference complete!"
fi

echo "=================================================================="
echo "📊 COMPUTING BENCHMARK METRICS ON 30 UNSEEN TEST CASES"
echo "=================================================================="

echo "🔍 Evaluating Config 12..."
$PYTHON evaluate_multiclass_test30_general.py "$PRED_DIR_12" "$BASE_DIR/config12_depthwise_method1_test30_metrics.csv"

echo "🔍 Evaluating Config 15..."
$PYTHON evaluate_multiclass_test30_general.py "$PRED_DIR_15" "$BASE_DIR/config15_depthwise_method1_siam_curriculum_test30_metrics.csv"

echo "=================================================================="
echo "🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
