#!/bin/bash
# push_to_local.sh
# -----------------
# Run this directly on the DGX to push any new results, models,
# or updates made on the DGX back to your local workstation.

LOCAL_HOST="ee23b009@10.23.12.88"
LOCAL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo " Pushing Updates from DGX to Local Workstation ($LOCAL_HOST)"
echo "=================================================================="

# 1. Push Model B2 results (Dataset106 Restored model)
echo -e "\n[1/3] Pushing Dataset106 model results to local PC..."
rsync -avP -e "ssh -o StrictHostKeyChecking=no" \
  $DGX_DIR/nnunet_data/nnUNet_results/Dataset106_UCSF_Saluja_Restored \
  $LOCAL_HOST:$LOCAL_DIR/nnunet_data/nnUNet_results/

# 2. Push any updated results or figures
echo -e "\n[2/3] Pushing results/ folder..."
rsync -avP -e "ssh -o StrictHostKeyChecking=no" \
  $DGX_DIR/results/ \
  $LOCAL_HOST:$LOCAL_DIR/results/

# 3. Push any updated markdown reports or scripts
echo -e "\n[3/3] Pushing updated reports and scripts..."
rsync -avP -e "ssh -o StrictHostKeyChecking=no" \
  $DGX_DIR/*.md \
  $DGX_DIR/*.py \
  $LOCAL_HOST:$LOCAL_DIR/

echo ""
echo "=================================================================="
echo " [✓] ALL UPDATES PUSHED TO LOCAL WORKSTATION!"
echo "=================================================================="
