#!/bin/bash
# run_predict_method3_siam_curriculum_multiclass_local.sh
# ---------------------------------------------------------
# Runs inference for Method 3 + SIAM + Curriculum BAA (Multiclass Dataset 102)
# on Local GPU 0, followed by test set metric evaluation.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=0

PRED_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_method3_siam_curriculum_multiclass_final"
mkdir -p "$PRED_DIR"

MODEL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_SIAM_CurriculumBAA__nnUNetPlans__3d_fullres/fold_0"

CHK="checkpoint_final.pth"
if [ ! -f "$MODEL_DIR/$CHK" ]; then
    echo "checkpoint_final.pth not found, using checkpoint_best.pth..."
    CHK="checkpoint_best.pth"
fi

echo "=========================================================="
echo "🚀 Running 3D Depthwise Separable + SIAM + Curriculum BAA Inference"
echo "Dataset: 102 (Dataset102_Restored - 30 Unseen Test Scans)"
echo "Model: nnUNetTrainer_Method3_SIAM_CurriculumBAA (1.49M params)"
echo "Checkpoint: $CHK"
echo "Output: $PRED_DIR"
echo "GPU: 0"
echo "=========================================================="

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict \
 -i /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs \
 -o "$PRED_DIR" \
 -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_SIAM_CurriculumBAA -chk "$CHK"

echo "=========================================================="
echo "🎯 Inference Complete! Running 30 Test Scan Evaluation..."
echo "=========================================================="
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3 /mnt/DATA/EE23B009/EE23B009_BTP/evaluate_method3_siam_curriculum_multiclass_test30.py "$PRED_DIR"

echo "=========================================================="
echo "✅ Method 3 Multiclass Evaluation Completed Successfully!"
echo "=========================================================="

