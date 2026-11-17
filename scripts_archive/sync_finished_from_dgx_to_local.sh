#!/bin/bash
# sync_finished_from_dgx_to_local.sh
# -----------------------------------------------------------------------------
# Pulls finished weights, logs, 3D predictions, and evaluation metrics for:
#   - Config 3  (Base nnU-Net + BAA Sensitive)
#   - Config 13 (Depthwise + Method 1 + SIAM)
#   - Config 9  (Depthwise + SIAM)
#   - Config 10 (Depthwise + BAA Sensitive)
#   - Config 14 (Depthwise + Method 1 + Curriculum BAA)
# from DGX to Local Workstation.
# -----------------------------------------------------------------------------

set -e

DGX_HOST="user@10.23.20.28"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"
LOCAL_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo "📥 Syncing All Finished Checkpoints & Metrics from DGX -> Local"
echo "=================================================================="

MODELS=(
    "nnUNetTrainer_BAA_Sensitive__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_InterLevel_SIAM__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA__nnUNetPlans__3d_fullres"
)

PREDS=(
    "predictions_config3_base_baa"
    "predictions_config13_depthwise_method1_siam"
    "predictions_config9_depthwise_siam"
    "predictions_config10_depthwise_baa"
    "predictions_config14_depthwise_method1_curriculum_baa"
)

CSVS=(
    "config3_base_baa_test30_metrics.csv"
    "config13_depthwise_method1_siam_test30_metrics.csv"
    "config9_depthwise_siam_test30_metrics.csv"
    "config10_depthwise_baa_test30_metrics.csv"
    "config14_depthwise_method1_curriculum_baa_test30_metrics.csv"
)

# 1. Sync Model Checkpoints
echo -e "\n📦 [1/3] Syncing Checkpoints, Plans & Training Logs..."
for m in "${MODELS[@]}"; do
    mkdir -p "$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset102_Restored/$m"
    rsync -avP --include="*/" \
               --include="checkpoint_*.pth" \
               --include="training_log_*.txt" \
               --include="*.json" \
               --exclude="*" \
               "$DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_results/Dataset102_Restored/$m/" \
               "$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset102_Restored/$m/" 2>/dev/null || true
done

# 2. Sync Predictions
echo -e "\n🔮 [2/3] Syncing 3D Prediction Masks (.nii.gz)..."
mkdir -p "$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset102_Restored"
for p in "${PREDS[@]}"; do
    rsync -avP "$DGX_HOST:$DGX_DIR/nnunet_data/nnUNet_results/Dataset102_Restored/$p" \
               "$LOCAL_DIR/nnunet_data/nnUNet_results/Dataset102_Restored/" 2>/dev/null || true
done

# 3. Sync Metrics CSVs
echo -e "\n📊 [3/3] Syncing Benchmark Metrics CSVs..."
for c in "${CSVS[@]}"; do
    rsync -avP "$DGX_HOST:$DGX_DIR/$c" "$LOCAL_DIR/" 2>/dev/null || true
done

echo ""
echo "=================================================================="
echo "✅ SYNC COMPLETED SUCCESSFULLY!"
echo "=================================================================="
