#!/bin/bash
# pull_from_workstation.sh
# -----------------------------------------------------------------------------
# Run this inside DGX (/Data3/EE23B009/EE23B009_BTP or ~):
# curl -s http://10.23.12.88:8080/pull_from_workstation.sh | bash
# -----------------------------------------------------------------------------
set -e

DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
if [ ! -d "$DGX_DIR" ]; then
    mkdir -p "$DGX_DIR"
fi
cd "$DGX_DIR"

SERVER="http://10.23.12.88:8080"
mkdir -p src training_logs src/losses src/models

echo "=================================================================="
echo "📥 Pulling all updated trainers, scripts, and GT labels from Workstation..."
echo "Source: $SERVER"
echo "Target: $DGX_DIR"
echo "=================================================================="

# Pull Real Ground Truth Test Labels (129 KB)
mkdir -p nnunet_data/nnUNet_raw/Dataset100_BraTS nnunet_data/nnUNet_raw/Dataset102_Restored
wget -q "$SERVER/labelsTs.tar.gz" -O labelsTs.tar.gz
tar -xzvf labelsTs.tar.gz -C nnunet_data/nnUNet_raw/Dataset100_BraTS/
cp -r nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs nnunet_data/nnUNet_raw/Dataset102_Restored/
rm -f labelsTs.tar.gz

# Trainers & Modules
wget -q "$SERVER/src/nnUNetTrainer_BAA_Sensitive.py" -O src/nnUNetTrainer_BAA_Sensitive.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_InterLevel_SIAM.py" -O src/nnUNetTrainer_Method3_InterLevel_SIAM.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_InterLevelAll.py" -O src/nnUNetTrainer_Method3_InterLevelAll.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA.py" -O src/nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_SIAM.py" -O src/nnUNetTrainer_Method3_SIAM.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_BAA_Sensitive.py" -O src/nnUNetTrainer_Method3_BAA_Sensitive.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA.py" -O src/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA.py
wget -q "$SERVER/src/nnUNetTrainer_Method3_SIAM_CurriculumBAA.py" -O src/nnUNetTrainer_Method3_SIAM_CurriculumBAA.py
wget -q "$SERVER/src/nnUNetTrainer_DepthwiseSeparable3D.py" -O src/nnUNetTrainer_DepthwiseSeparable3D.py
wget -q "$SERVER/src/nnUNetTrainer_InterLevel_SIAM.py" -O src/nnUNetTrainer_InterLevel_SIAM.py
wget -q "$SERVER/src/losses/baa_loss.py" -O src/losses/baa_loss.py
wget -q "$SERVER/src/losses/boundary_baa_loss.py" -O src/losses/boundary_baa_loss.py
wget -q "$SERVER/src/models/siam.py" -O src/models/siam.py

# Launch Scripts (Only active binary configs for DGX GPUs 3 & 4)
wget -q "$SERVER/launch_dgx_binary_config9_depthwise_siam.sh" -O launch_dgx_binary_config9_depthwise_siam.sh
wget -q "$SERVER/launch_dgx_binary_config10_depthwise_baa.sh" -O launch_dgx_binary_config10_depthwise_baa.sh
wget -q "$SERVER/start_dgx_binary_9_and_10.sh" -O start_dgx_binary_9_and_10.sh

# Fast Inference & Eval Script on DGX
wget -q "$SERVER/evaluate_saluja_test30.py" -O evaluate_saluja_test30.py

chmod +x launch_dgx_*.sh start_dgx_*.sh evaluate_*.py 2>/dev/null || true

echo "📦 Installing trainers into DGX python environment..."
for ENV_PATH in "/Data3/EE23B009/brats_env" "/Data3/EE23B009/EE23B009_BTP/brats_env" "$HOME/brats_env"; do
    if [ -d "$ENV_PATH/lib" ]; then
        cp src/nnUNetTrainer_*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/ 2>/dev/null || true
        mkdir -p "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses 2>/dev/null || true
        mkdir -p "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models 2>/dev/null || true
        cp src/losses/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/losses/ 2>/dev/null || true
        cp src/models/*.py "$ENV_PATH"/lib/python*/site-packages/nnunetv2/training/nnUNetTrainer/models/ 2>/dev/null || true
    fi
done

echo ""
echo "=================================================================="
echo "✅ ALL FILES & GROUND TRUTH LABELS INSTALLED ON DGX SUCCESSFULLY!"
echo "=================================================================="
