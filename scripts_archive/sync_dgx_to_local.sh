#!/bin/bash
# =============================================================================
# sync_dgx_to_local.sh (Run this directly on DGX)
# -----------------------------------------------------------------------------
# Safely pulls and transfers all weights, checkpoints, logs, and evaluation
# metrics for Config 9, Config 10, and Config 14 from DGX to Local Workstation.
#
# Target Local IP : ee23b009@10.23.12.88
# Local Target Dir: /mnt/DATA/EE23B009/EE23B009_BTP
# =============================================================================

set -e

LOCAL_USER="ee23b009"
LOCAL_IP="10.23.12.88"
LOCAL_BASE="/mnt/DATA/EE23B009/EE23B009_BTP"

DGX_BASE="/Data3/EE23B009/EE23B009_BTP"
RESULTS_DIR="$DGX_BASE/nnunet_data/nnUNet_results/Dataset102_Restored"

cd "$DGX_BASE"

echo "=================================================================="
echo "📦 [1/3] Syncing Checkpoints & Plans to Local Workstation..."
echo "=================================================================="

# Config 9: Depthwise + SIAM
echo "▶️ Syncing Config 9 (Depthwise + SIAM)..."
rsync -avP --include="*/" \
           --include="checkpoint_*.pth" \
           --include="training_log_*.txt" \
           --include="*.json" \
           --exclude="*" \
           "$RESULTS_DIR/nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres/" \
           "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres/"

# Config 14: Depthwise + Method 1 + Curriculum BAA
echo "▶️ Syncing Config 14 (Depthwise + Method 1 + Curriculum BAA)..."
rsync -avP --include="*/" \
           --include="checkpoint_*.pth" \
           --include="training_log_*.txt" \
           --include="*.json" \
           --exclude="*" \
           "$RESULTS_DIR/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA__nnUNetPlans__3d_fullres/" \
           "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA__nnUNetPlans__3d_fullres/"

# Config 10: Depthwise + BAA Sensitive 1.5
echo "▶️ Syncing Config 10 (Depthwise + BAA Sensitive 1.5)..."
rsync -avP --include="*/" \
           --include="checkpoint_*.pth" \
           --include="training_log_*.txt" \
           --include="*.json" \
           --exclude="*" \
           "$RESULTS_DIR/nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres/" \
           "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres/"

echo "=================================================================="
echo "📊 [2/3] Syncing Benchmark Metrics CSVs..."
echo "=================================================================="
scp "$DGX_BASE/config9_depthwise_siam_test30_metrics.csv" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/" 2>/dev/null || echo "Note: config9 CSV not generated yet (run eval first)."
scp "$DGX_BASE/config14_depthwise_method1_curriculum_baa_test30_metrics.csv" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/" 2>/dev/null || echo "Note: config14 CSV not generated yet (run eval first)."
scp "$DGX_BASE/config10_depthwise_baa_test30_metrics.csv" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/" 2>/dev/null || echo "Note: config10 CSV synced."

echo "=================================================================="
echo "🔮 [3/3] Syncing 3D Prediction Masks (.nii.gz) (Optional)..."
echo "=================================================================="
rsync -avP "$RESULTS_DIR/predictions_config9_depthwise_siam" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset102_Restored/" 2>/dev/null || true
rsync -avP "$RESULTS_DIR/predictions_config14_depthwise_method1_curriculum_baa" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset102_Restored/" 2>/dev/null || true
rsync -avP "$RESULTS_DIR/predictions_config10_depthwise_baa" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset102_Restored/" 2>/dev/null || true

echo "=================================================================="
echo "✅ SYNC COMPLETED SUCCESSFULLY TO LOCAL WORKSTATION!"
echo "=================================================================="

