#!/bin/bash
# prepare_dataset102_dgx.sh
# -------------------------
# Pulls exact plans, fingerprints, and splits for Dataset102_Restored from local machine,
# then runs high-throughput parallel preprocessing directly from DGX's raw images.

set -e

export nnUNet_raw="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

TARGET_DIR="$nnUNet_preprocessed/Dataset102_Restored"
mkdir -p "$TARGET_DIR"

echo "=================================================================="
echo "🔄 [1/2] Syncing Dataset102 exact plans & splits (< 20 MB, ~3 sec)..."
echo "=================================================================="

rsync -avP ee23b009@10.23.12.88:/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed/Dataset102_Restored/'*.json' \
      "$TARGET_DIR/"

rsync -avP ee23b009@10.23.12.88:/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed/Dataset102_Restored/gt_segmentations/ \
      "$TARGET_DIR/gt_segmentations/"

echo ""
echo "=================================================================="
echo "⚡ [2/2] Running 3d_fullres preprocessing on DGX (16 workers, ~3 min)..."
echo "=================================================================="

if [ -f "/Data3/EE23B009/brats_env/bin/nnUNetv2_preprocess" ]; then
    /Data3/EE23B009/brats_env/bin/nnUNetv2_preprocess -d 102 -c 3d_fullres -np 16
elif [ -f "/Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_preprocess" ]; then
    /Data3/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_preprocess -d 102 -c 3d_fullres -np 16
else
    nnUNetv2_preprocess -d 102 -c 3d_fullres -np 16
fi

echo ""
echo "=================================================================="
echo "🎉 Dataset102_Restored is 100% PREPROCESSED & READY ON DGX!"
echo "=================================================================="

