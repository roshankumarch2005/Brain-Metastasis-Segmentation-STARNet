#!/bin/bash
# sync_to_dgx.sh
# ----------------
# Pushes all results, models, datasets, trainers, and documentation from local machine to DGX server.

DGX_HOST="user@10.23.20.28"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
LOCAL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo " 🔄 SYNCING EVERYTHING FROM LOCAL WORKSTATION TO DGX ($DGX_HOST)"
echo " DGX Directory: $DGX_DIR"
echo "=================================================================="

# 1. Ensure remote target directories exist
echo -e "\n[1/6] Creating remote directories on DGX..."
ssh $DGX_HOST "mkdir -p $DGX_DIR/nnunet_data/nnUNet_results $DGX_DIR/nnunet_data/nnUNet_raw $DGX_DIR/nnunet_data/nnUNet_preprocessed $DGX_DIR/src $DGX_DIR/results"

# 2. Sync results/ (All 3D HTMLs, B0, B1, B3, M9, M10 viewers, PDF and HTML reports)
echo -e "\n[2/6] Transferring results/ directory..."
rsync -avP $LOCAL_DIR/results/ $DGX_HOST:$DGX_DIR/results/

# 3. Sync source code, training logs, reports, and scripts
echo -e "\n[3/6] Transferring key reports (BTP_Final_Report.md, Literature_vs_Our_BTP_Results_Comparison.md, Methods_1_2_3.md), source code, and scripts..."
rsync -avP $LOCAL_DIR/src/ $DGX_HOST:$DGX_DIR/src/
rsync -avP $LOCAL_DIR/training_logs/ $DGX_HOST:$DGX_DIR/training_logs/ 2>/dev/null || true
rsync -avP $LOCAL_DIR/*.md \
           $LOCAL_DIR/*.py \
           $LOCAL_DIR/*.sh \
           $LOCAL_DIR/*.csv \
           $LOCAL_DIR/*.pptx \
           $LOCAL_DIR/*.pdf \
           $DGX_HOST:$DGX_DIR/

# 4. Sync finalized nnUNet model weights, test predictions, and logs
echo -e "\n[4/6] Transferring nnUNet trained model weights & logs (Dataset102, Dataset105, Dataset100)..."
rsync -avP $LOCAL_DIR/nnunet_data/nnUNet_results/Dataset102_Restored \
           $LOCAL_DIR/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact \
           $LOCAL_DIR/nnunet_data/nnUNet_results/Dataset100_BraTS \
           $DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_results/

# 5. Sync Dataset105 raw and preprocessed data (required for Model 2 training on DGX)
echo -e "\n[5/6] Verifying/Transferring Dataset105 raw & preprocessed data for DGX training..."
rsync -avP $LOCAL_DIR/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/ \
           $DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/

rsync -avP $LOCAL_DIR/nnunet_data/nnUNet_preprocessed/Dataset105_UCSF_Saluja_Exact/ \
           $DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_preprocessed/Dataset105_UCSF_Saluja_Exact/

# 6. Install custom trainers into DGX python environment
echo -e "\n[6/6] Installing custom trainers into DGX nnUNet package..."
ssh $DGX_HOST "cp $DGX_DIR/src/nnUNetTrainer_InterLevel*.py $DGX_DIR/brats_env/lib/python3.8/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || cp $DGX_DIR/src/nnUNetTrainer_InterLevel*.py ~/brats_env/lib/python3.8/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true"

echo ""
echo "=================================================================="
echo " [✓] ALL RESULTS, MODELS, DATASETS & TRAINERS SYNCED TO DGX!"
echo "=================================================================="
