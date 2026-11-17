#!/bin/bash
# run_eval_4sweeps_test30.sh
# -----------------------------------------------------------------
# Runs 3D fullres inference on 30 unseen test cases across all 4 M10 sweeps
# simultaneously on GPUs 0, 1, 2, 3, then computes Dice and HD95 metrics.

set -e

export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"

INPUT_DIR="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs"
BASE_RES="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored"
PY="/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3"
PRED_BIN="/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict"

mkdir -p "$BASE_RES/preds_test30_m10_lam05_tr20"
mkdir -p "$BASE_RES/preds_test30_m10_lam15_tr20"
mkdir -p "$BASE_RES/preds_test30_m10_lam10_tr10"
mkdir -p "$BASE_RES/preds_test30_m10_lam10_tr30"

echo "=========================================================="
echo "🚀 Launching 30-Case Test Inference across 4 GPUs in Parallel"
echo "Time: $(date)"
echo "=========================================================="

# GPU 0: Lam05_Tr20
CUDA_VISIBLE_DEVICES=0 "$PRED_BIN" \
  -i "$INPUT_DIR" -o "$BASE_RES/preds_test30_m10_lam05_tr20" \
  -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_M10_Lam05_Tr20 -chk checkpoint_final.pth \
  > "$BASE_RES/pred_gpu0_lam05_tr20.log" 2>&1 &
PID0=$!

# GPU 1: Lam15_Tr20
CUDA_VISIBLE_DEVICES=1 "$PRED_BIN" \
  -i "$INPUT_DIR" -o "$BASE_RES/preds_test30_m10_lam15_tr20" \
  -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_M10_Lam15_Tr20 -chk checkpoint_final.pth \
  > "$BASE_RES/pred_gpu1_lam15_tr20.log" 2>&1 &
PID1=$!

# GPU 2: Lam10_Tr10
CUDA_VISIBLE_DEVICES=2 "$PRED_BIN" \
  -i "$INPUT_DIR" -o "$BASE_RES/preds_test30_m10_lam10_tr10" \
  -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_M10_Lam10_Tr10 -chk checkpoint_final.pth \
  > "$BASE_RES/pred_gpu2_lam10_tr10.log" 2>&1 &
PID2=$!

# GPU 3: Lam10_Tr30
CUDA_VISIBLE_DEVICES=3 "$PRED_BIN" \
  -i "$INPUT_DIR" -o "$BASE_RES/preds_test30_m10_lam10_tr30" \
  -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer_M10_Lam10_Tr30 -chk checkpoint_final.pth \
  > "$BASE_RES/pred_gpu3_lam10_tr30.log" 2>&1 &
PID3=$!

echo "Inference jobs launched:"
echo "  GPU 0 (PID $PID0): Lam05_Tr20"
echo "  GPU 1 (PID $PID1): Lam15_Tr20"
echo "  GPU 2 (PID $PID2): Lam10_Tr10"
echo "  GPU 3 (PID $PID3): Lam10_Tr30"

# Wait for all 4
wait $PID0
echo "✅ GPU 0 (Lam05_Tr20) finished!"

wait $PID1
echo "✅ GPU 1 (Lam15_Tr20) finished!"

wait $PID2
echo "✅ GPU 2 (Lam10_Tr10) finished!"

wait $PID3
echo "✅ GPU 3 (Lam10_Tr30) finished!"

echo "=========================================================="
echo "🎯 All 4 inferences complete! Calculating Dice and HD95..."
echo "=========================================================="

