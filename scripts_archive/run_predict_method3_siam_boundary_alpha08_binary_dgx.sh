#!/bin/bash
# run_predict_method3_siam_boundary_alpha08_binary_dgx.sh
# -----------------------------------------------------------
# Inference and evaluation for Method 3 + SIAM + Boundary Alpha08 (Binary B3 engine)
# on DGX GPU 5 (Dataset 105 UCSF Saluja Exact - 30 Unseen Test Scans)

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=5

PRED_DIR="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_method3_siam_boundary_alpha08_final"
mkdir -p "$PRED_DIR"

MODEL_DIR="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/nnUNetTrainer_Method3_SIAM_Boundary_Alpha08__nnUNetPlans__3d_fullres/fold_0"

CHK="checkpoint_final.pth"
if [ ! -f "$MODEL_DIR/$CHK" ]; then
    echo "checkpoint_final.pth not found, using checkpoint_best.pth..."
    CHK="checkpoint_best.pth"
fi

echo "=========================================================="
echo "🚀 Running Test Inference: Method 3 + SIAM + Boundary Alpha08"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact - 30 Unseen Test Scans)"
echo "Trainer: nnUNetTrainer_Method3_SIAM_Boundary_Alpha08"
echo "Model Dir: $MODEL_DIR"
echo "Checkpoint: $CHK"
echo "Output: $PRED_DIR"
echo "GPU: 5"
echo "=========================================================="

/Data3/EE23B009/brats_env/bin/nnUNetv2_predict \
 -i /Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/imagesTs \
 -o "$PRED_DIR" \
 -d 105 -c 3d_fullres -f 0 -tr nnUNetTrainer_Method3_SIAM_Boundary_Alpha08 -chk "$CHK"

echo "=========================================================="
echo "🎯 Inference Complete! Running Official 30 Test Scan Evaluation..."
echo "=========================================================="
/Data3/EE23B009/brats_env/bin/python3 /Data3/EE23B009/EE23B009_BTP/evaluate_saluja_test30.py "$PRED_DIR" | tee /Data3/EE23B009/EE23B009_BTP/method3_siam_boundary_alpha08_test_metrics.txt

echo "=========================================================="
echo "✅ Method 3 Binary Evaluation Completed Successfully!"
echo "=========================================================="

