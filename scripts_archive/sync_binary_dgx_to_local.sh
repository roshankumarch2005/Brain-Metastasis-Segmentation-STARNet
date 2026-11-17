#!/bin/bash
# =============================================================================
# sync_binary_dgx_to_local.sh (Run this directly on DGX)
# -----------------------------------------------------------------------------
# Safely transfers all weights, checkpoints, logs, predictions, and evaluation
# results for Binary Configurations (Configs 12, 13, 14, and optional 9, 10, 11, 15)
# from DGX to Local Workstation.
#
# Target Local IP : ee23b009@10.23.12.88
# Local Target Dir: /mnt/DATA/EE23B009/EE23B009_BTP
# =============================================================================

set -e

LOCAL_USER="ee23b009"
LOCAL_IP="10.23.12.88"
LOCAL_BASE="/mnt/DATA/EE23B009/EE23B009_BTP"

DGX_BASE="/Data3/EE23B009/EE23B009_BTP"
RESULTS_DIR="$DGX_BASE/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact"
LOCAL_RESULTS_DIR="$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact"

cd "$DGX_BASE"

echo "=================================================================="
echo "📦 [1/3] Syncing Checkpoints & Training Logs from DGX -> Local"
echo "=================================================================="

MODELS=(
    "nnUNetTrainer_Method3_InterLevelAll__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_InterLevel_SIAM__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_SIAM__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_BAA_Sensitive__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_SIAM_CurriculumBAA__nnUNetPlans__3d_fullres"
    "nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA__nnUNetPlans__3d_fullres"
)

for m in "${MODELS[@]}"; do
    if [ -d "$RESULTS_DIR/$m" ]; then
        echo "▶️ Syncing model checkpoint: $m ..."
        rsync -avP --include="*/" \
                   --include="checkpoint_*.pth" \
                   --include="training_log_*.txt" \
                   --include="*.json" \
                   --exclude="*" \
                   "$RESULTS_DIR/$m/" \
                   "$LOCAL_USER@$LOCAL_IP:$LOCAL_RESULTS_DIR/$m/"
    fi
done

echo -e "\n=================================================================="
echo "📝 [2/3] Syncing Training & Evaluation Logs..."
echo "=================================================================="
rsync -avP "$DGX_BASE/training_logs/train_dgx_binary_*.log" "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/training_logs/" 2>/dev/null || true
rsync -avP "$DGX_BASE"/config*_eval.txt "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/" 2>/dev/null || true
rsync -avP "$DGX_BASE"/*binary*test30*.csv "$LOCAL_USER@$LOCAL_IP:$LOCAL_BASE/" 2>/dev/null || true

echo -e "\n=================================================================="
echo "🔮 [3/3] Syncing Predictions (if generated)..."
echo "=================================================================="
rsync -avP "$RESULTS_DIR"/predictions_config* "$LOCAL_USER@$LOCAL_IP:$LOCAL_RESULTS_DIR/" 2>/dev/null || true

echo -e "\n=================================================================="
echo "✅ DGX -> Local Sync Complete!"
echo "=================================================================="
