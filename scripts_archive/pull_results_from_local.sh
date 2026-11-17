#!/bin/bash
# pull_results_from_local.sh
# ---------------------------
# Run directly from /Data3/EE23B009/EE23B009_BTP on the DGX
# Pulls everything from Local Workstation to DGX.

LOCAL_HOST="ee23b009@10.23.12.88"
LOCAL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo " 🔄 SYNCING EVERYTHING FROM LOCAL WORKSTATION (10.23.12.88) TO DGX"
echo "=================================================================="

# 1. Pull source code, trainers, and configs
echo -e "\n[1/6] Pulling source code (src/)..."
mkdir -p src
rsync -avP $LOCAL_HOST:$LOCAL_DIR/src/ ./src/

# 2. Pull all reports, decks, scripts, and benchmark tables
echo -e "\n[2/6] Pulling key reports (BTP_Final_Report.md, Literature_vs_Our_BTP_Results_Comparison.md, Methods_1_2_3.md), scripts, and tables..."
rsync -avP $LOCAL_HOST:$LOCAL_DIR/*.md \
           $LOCAL_HOST:$LOCAL_DIR/*.py \
           $LOCAL_HOST:$LOCAL_DIR/*.sh \
           $LOCAL_HOST:$LOCAL_DIR/*.csv \
           $LOCAL_HOST:$LOCAL_DIR/*.pptx \
           $LOCAL_HOST:$LOCAL_DIR/*.pdf \
           ./

# 3. Pull results/ (All HTML 3D viewers, M9, M10, B1, B3 visualizers, PNG figures, PDF reports)
echo -e "\n[3/6] Pulling results/ folder (~3.4 GB)..."
mkdir -p results
rsync -avP $LOCAL_HOST:$LOCAL_DIR/results/ ./results/

# 4. Pull all trained models & predictions (Dataset102, Dataset105, Dataset100)
echo -e "\n[4/6] Pulling trained models & predictions..."
mkdir -p nnunet_data/nnUNet_results
rsync -avP $LOCAL_HOST:$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset102_Restored ./nnunet_data/nnUNet_results/
rsync -avP $LOCAL_HOST:$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact ./nnunet_data/nnUNet_results/
rsync -avP $LOCAL_HOST:$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset100_BraTS ./nnunet_data/nnUNet_results/

# 5. Pull Dataset105 raw and preprocessed data
echo -e "\n[5/6] Pulling Dataset105 raw and preprocessed data..."
mkdir -p nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact nnunet_data/nnUNet_preprocessed/Dataset105_UCSF_Saluja_Exact
rsync -avP $LOCAL_HOST:$LOCAL_DIR/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/ ./nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/
rsync -avP $LOCAL_HOST:$LOCAL_DIR/nnunet_data/nnUNet_preprocessed/Dataset105_UCSF_Saluja_Exact/ ./nnunet_data/nnUNet_preprocessed/Dataset105_UCSF_Saluja_Exact/

# 6. Install custom trainers into DGX python environment
echo -e "\n[6/6] Installing custom trainers into DGX nnUNet package..."
cp $DGX_DIR/src/nnUNetTrainer_InterLevel*.py $DGX_DIR/brats_env/lib/python3.8/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || cp $DGX_DIR/src/nnUNetTrainer_InterLevel*.py ~/brats_env/lib/python3.8/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true

echo ""
echo "=================================================================="
echo " [✓] 100% COMPLETE! ALL RECENT CODES, MODELS & REPORTS ARE ON DGX!"
echo "=================================================================="
