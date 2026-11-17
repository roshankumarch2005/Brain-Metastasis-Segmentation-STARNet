#!/bin/bash
set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=1

cd /mnt/DATA/EE23B009/EE23B009_BTP
PYTHON="/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python"

echo "=========================================================="
echo "Dataset 106 Preprocessing (3d_fullres) using 16 CPU workers"
echo "=========================================================="
$PYTHON -c "
from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess
preprocess([106], configurations=('3d_fullres',), num_processes=(16,))
print('Preprocessing 3d_fullres completed successfully!')
"

echo "=========================================================="
echo "Launching SIAM Attention + BAA 1.5 Training on GPU 1"
echo "Dataset: 106 (Saluja Exact Modalities + N4/NLM Restored)"
echo "Target: Binary Enhancing Metastasis"
echo "Trainer: nnUNetTrainerSIAM_BAA_Sensitive"
echo "=========================================================="
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 106 3d_fullres 0 -tr nnUNetTrainerSIAM_BAA_Sensitive
