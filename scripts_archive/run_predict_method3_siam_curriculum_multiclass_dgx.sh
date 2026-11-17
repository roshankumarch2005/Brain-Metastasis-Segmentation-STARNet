#!/bin/bash
# run_predict_method3_siam_curriculum_multiclass_dgx.sh
# --------------------------------------------------------
# Inference and evaluation for Method 3 + SIAM + Curriculum BAA (Multiclass M8 engine)
# on DGX GPU 6 (Dataset 102 Restored - 30 Unseen Test Scans)

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=6

PRED_DIR="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_method3_siam_curriculum_multiclass_final"
mkdir -p "$PRED_DIR"

MODEL_DIR="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_SIAM_CurriculumBAA__nnUNetPlans__3d_fullres/fold_0"

CHK="checkpoint_final.pth"
if [ ! -f "$MODEL_DIR/$CHK" ]; then
    echo "checkpoint_final.pth not found, using checkpoint_best.pth..."
    CHK="checkpoint_best.pth"
fi

echo "=========================================================="
echo "🚀 Running Test Inference: Method 3 + SIAM + Curriculum BAA"
echo "Dataset: 102 (Dataset102_Restored - 30 Unseen Test Scans)"
echo "Trainer: nnUNetTrainer_Method3_SIAM_CurriculumBAA"
echo "Model Dir: $MODEL_DIR"
echo "Checkpoint: $CHK"
echo "Output: $PRED_DIR"
echo "GPU: 6"
echo "=========================================================="

/Data3/EE23B009/brats_env/bin/nnUNetv2_predict \
 -i /Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs \
 -o "$PRED_DIR" \
 -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_SIAM_CurriculumBAA -chk "$CHK"

echo "=========================================================="
echo "🎯 Inference Complete! Running 30 Test Scan Evaluation..."
echo "=========================================================="
/Data3/EE23B009/brats_env/bin/python3 /Data3/EE23B009/EE23B009_BTP/evaluate_depthwise_multiclass_test30.py "$PRED_DIR" | tee /Data3/EE23B009/EE23B009_BTP/method3_siam_curriculum_multiclass_test_metrics.txt

echo "=========================================================="
echo "✅ Method 3 Multiclass Evaluation Completed Successfully!"
echo "=========================================================="

