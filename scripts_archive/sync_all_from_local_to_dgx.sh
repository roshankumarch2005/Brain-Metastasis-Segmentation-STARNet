#!/bin/bash
# sync_all_from_local_to_dgx.sh
# --------------------------------
# Run this on DGX (user@eeiitt:/Data3/EE23B009/EE23B009_BTP)
# Pulls all updated markdown reports, source trainers, loss functions, scripts,
# and dashboard results from Local Station (10.23.12.88).
#
# STRICT EXCLUSIONS:
#   - nnUNetTrainer_Decoder1_HalfRes (Currently actively training on Local GPU 0!)
#   - brats_env/ (Environment specific)
#   - *.pth (Checkpoints)
#   - raw / preprocessed datasets (Already on DGX)

set -e

LOCAL_USER="ee23b009"
LOCAL_IP="10.23.12.88"
LOCAL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo "🔄 PULLING UPDATES FROM LOCAL TO DGX ($LOCAL_IP -> $DGX_DIR)"
echo "Excluding active training: nnUNetTrainer_Decoder1_HalfRes (Local GPU 0)"
echo "=================================================================="

cd "$DGX_DIR"

rsync -avP \
  --exclude='data/' \
  --exclude='data_downloads/' \
  --exclude='results/' \
  --exclude='smp_env/' \
  --exclude='brats_env/' \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pth' \
  --exclude='*.nii.gz' \
  --exclude='nnunet_data/' \
  "${LOCAL_USER}@${LOCAL_IP}:${LOCAL_DIR}/src/" "$DGX_DIR/src/"

rsync -avP \
  "${LOCAL_USER}@${LOCAL_IP}:${LOCAL_DIR}/*.md" \
  "${LOCAL_USER}@${LOCAL_IP}:${LOCAL_DIR}/*.sh" \
  "${LOCAL_USER}@${LOCAL_IP}:${LOCAL_DIR}/*.py" \
  "$DGX_DIR/"

echo -e "\n[✓] Installing new trainers into DGX python environment..."
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp -u "$DGX_DIR"/src/nnUNetTrainer_*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        cp -u "$DGX_DIR"/src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
        cp -u "$DGX_DIR"/src/models/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models/ 2>/dev/null || true
    fi
done

chmod +x "$DGX_DIR"/*.sh

echo ""
echo "=================================================================="
echo "🎉 SYNC COMPLETE! All reports, trainers, and launch scripts ready."
echo "=================================================================="

