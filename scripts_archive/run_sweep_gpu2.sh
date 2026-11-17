#!/bin/bash
export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=2

cd /mnt/DATA/EE23B009/EE23B009_BTP
mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/sweep_logs

CKPT="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainer_M10_Lam10_Tr10__nnUNetPlans__3d_fullres/fold_0/checkpoint_latest.pth"
CONT_FLAG=""
if [ -f "$CKPT" ]; then
    CONT_FLAG="--c"
fi

echo "=========================================================="
echo "🚀 [GPU 2] Starting M10 Sweep: lambda = 1.00, Tr = 0.10 (Resume: ${CONT_FLAG:-No})"
echo "=========================================================="
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainer_M10_Lam10_Tr10 $CONT_FLAG 2>&1 | tee -a /mnt/DATA/EE23B009/EE23B009_BTP/sweep_logs/sweep_gpu2.log
