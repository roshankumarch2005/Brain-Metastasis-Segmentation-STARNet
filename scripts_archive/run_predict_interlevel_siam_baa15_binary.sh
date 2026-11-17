#!/bin/bash
# run_predict_interlevel_siam_baa15_binary.sh
# --------------------------------------------
# Evaluates Model 2: Method 1 + SIAM + BAA Sensitive (beta_scale=1.5)
# on the 30 unseen test scans of Dataset105_UCSF_Saluja_Exact using Local GPU 2.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=2

PRED_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_interlevel_siam_baa15_final"
mkdir -p "$PRED_DIR"

echo "=========================================================="
echo "🚀 Running Test Inference: Method 1 + SIAM + BAA Sensitive (1.5)"
echo "Dataset: 105 (Dataset105_UCSF_Saluja_Exact - 30 Unseen Test Scans)"
echo "Trainer: nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive"
echo "Output: $PRED_DIR"
echo "GPU: 2"
echo "=========================================================="

# Check whether checkpoint_best.pth or checkpoint_final.pth exists
MODEL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive__nnUNetPlans__3d_fullres/fold_0"

CHK="checkpoint_final.pth"
if [ ! -f "$MODEL_DIR/$CHK" ]; then
    CHK="checkpoint_best.pth"
fi

echo "Using checkpoint: $CHK"

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict \
 -i /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/imagesTs \
 -o "$PRED_DIR" \
 -d 105 -c 3d_fullres -f 0 -tr nnUNetTrainer_InterLevel_SIAM_BAA_Sensitive -chk "$CHK"

echo "Inference complete! Now running official test set evaluation..."
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3 /mnt/DATA/EE23B009/EE23B009_BTP/evaluate_saluja_test30.py "$PRED_DIR" | tee /mnt/DATA/EE23B009/EE23B009_BTP/interlevel_siam_baa15_test_metrics.txt

echo "ALL DONE!"

