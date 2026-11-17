#!/bin/bash
# sync_from_dgx.sh
# -----------------
# Run this on your LOCAL WORKSTATION to pull EVERYTHING from DGX to Local PC:
# - All model weights (checkpoint_best.pth, checkpoint_final.pth)
# - All training logs
# - All test predictions (.nii.gz)
# - All evaluation results, 3D dashboards, and markdown reports

DGX_HOST="user@10.23.20.28"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
LOCAL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo " 🔄 COMPLETE SYNC: Pulling ALL Models & Results from DGX -> Local"
echo " DGX Source: $DGX_HOST:$DGX_DIR"
echo " Local Dest: $LOCAL_DIR"
echo "=================================================================="

# 1. Pull Model Weights & Logs (Dataset100 BraTS RAW, Dataset106 Restored)
echo -e "\n[1/4] Pulling Trained Model Checkpoints & Logs from DGX..."
mkdir -p $LOCAL_DIR/nnunet_data/nnUNet_results
rsync -avP --include="*/" \
           --include="checkpoint_best.pth" \
           --include="checkpoint_final.pth" \
           --include="training_log_*.txt" \
           --include="plans.json" \
           --include="dataset.json" \
           --exclude="*" \
           $DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_results/Dataset100_BraTS/ \
           $LOCAL_DIR/nnunet_data/nnUNet_results/Dataset100_BraTS/

rsync -avP --include="*/" \
           --include="checkpoint_best.pth" \
           --include="checkpoint_final.pth" \
           --include="training_log_*.txt" \
           --include="plans.json" \
           --include="dataset.json" \
           --exclude="*" \
           $DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_results/Dataset106_UCSF_Saluja_Restored/ \
           $LOCAL_DIR/nnunet_data/nnUNet_results/Dataset106_UCSF_Saluja_Restored/

# 2. Pull All Test Prediction Folders
echo -e "\n[2/4] Pulling Test Predictions (3D NIfTI masks)..."
rsync -avP $DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_results/predictions_test_m5_raw \
           $LOCAL_DIR/nnunet_data/nnUNet_results/

# 3. Pull results/ folder (3D HTML dashboards, PDFs, PNG figures)
echo -e "\n[3/4] Pulling results/ directory..."
rsync -avP $DGX_HOST:$DGX_DIR/results/ \
           $LOCAL_DIR/results/

# 4. Pull any updated scripts, CSV benchmarks, and markdown reports
echo -e "\n[4/4] Pulling updated reports, CSV benchmarks, and scripts..."
rsync -avP $DGX_HOST:$DGX_DIR/*.md \
           $DGX_HOST:$DGX_DIR/*.py \
           $DGX_HOST:$DGX_DIR/*.csv \
           $LOCAL_DIR/

echo ""
echo "=================================================================="
echo " [✓] 100% COMPLETE: ALL WEIGHTS, PREDICTIONS & LOGS ARE IN SYNC!"
echo "=================================================================="
