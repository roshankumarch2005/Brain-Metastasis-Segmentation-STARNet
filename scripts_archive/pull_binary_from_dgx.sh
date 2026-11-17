#!/bin/bash
# =============================================================================
# pull_binary_from_dgx.sh (Run this directly on Local Workstation)
# -----------------------------------------------------------------------------
# Pulls checkpoints, logs, and evaluation files from DGX to Local Workstation.
# Prompts for DGX password (user@10.23.20.28).
# =============================================================================

set -e

DGX_USER="user"
DGX_IP="10.23.20.28"
DGX_BASE="/Data3/EE23B009/EE23B009_BTP"

LOCAL_BASE="/mnt/DATA/EE23B009/EE23B009_BTP"
RESULTS_DIR="$DGX_BASE/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact"
LOCAL_RESULTS_DIR="$LOCAL_BASE/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact"

mkdir -p "$LOCAL_RESULTS_DIR"
mkdir -p "$LOCAL_BASE/training_logs"

echo "=================================================================="
echo "📥 Pulling Binary Configs Checkpoints & Logs from DGX -> Local"
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
    echo "▶️ Checking and pulling $m..."
    mkdir -p "$LOCAL_RESULTS_DIR/$m"
    rsync -avP --include="*/" \
               --include="checkpoint_*.pth" \
               --include="training_log_*.txt" \
               --include="*.json" \
               --exclude="*" \
               "$DGX_USER@$DGX_IP:$RESULTS_DIR/$m/" \
               "$LOCAL_RESULTS_DIR/$m/" 2>/dev/null || true
done

echo -e "\n=================================================================="
echo "📝 Pulling Logs & Evaluation Summaries..."
echo "=================================================================="
rsync -avP "$DGX_USER@$DGX_IP:$DGX_BASE/training_logs/train_dgx_binary_*.log" "$LOCAL_BASE/training_logs/" 2>/dev/null || true
rsync -avP "$DGX_USER@$DGX_IP:$DGX_BASE/config*_eval.txt" "$LOCAL_BASE/" 2>/dev/null || true
rsync -avP "$DGX_USER@$DGX_IP:$DGX_BASE/*binary*test30*.csv" "$LOCAL_BASE/" 2>/dev/null || true

echo -e "\n=================================================================="
echo "🔮 Pulling Predictions (if generated)..."
echo "=================================================================="
rsync -avP "$DGX_USER@$DGX_IP:$RESULTS_DIR/predictions_config*" "$LOCAL_RESULTS_DIR/" 2>/dev/null || true

echo -e "\n=================================================================="
echo "✅ Done! All binary artifacts pulled to local."
echo "=================================================================="
