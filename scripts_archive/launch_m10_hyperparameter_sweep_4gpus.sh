#!/bin/bash
set -e

BASE_DIR="/mnt/DATA/EE23B009/EE23B009_BTP"
export nnUNet_raw="$BASE_DIR/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="$BASE_DIR/nnunet_data/nnUNet_results"

PYTHON_BIN="$BASE_DIR/brats_env/bin/nnUNetv2_train"

mkdir -p "$BASE_DIR/logs_sweeps"

echo "=========================================================================="
echo "🚀 LAUNCHING 4-GPU MODEL M10 HYPERPARAMETER SWEEP ON DATASET 102"
echo "=========================================================================="
echo "GPU 0: nnUNetTrainer_M10_Lam05_Tr20  (lambda = 0.50, Tr = 0.20)"
echo "GPU 1: nnUNetTrainer_M10_Lam15_Tr20  (lambda = 1.50, Tr = 0.20)"
echo "GPU 2: nnUNetTrainer_M10_Lam10_Tr10  (lambda = 1.00, Tr = 0.10)"
echo "GPU 3: nnUNetTrainer_M10_Lam10_Tr30  (lambda = 1.00, Tr = 0.30)"
echo "=========================================================================="

# GPU 0: lambda = 0.50, Tr = 0.20
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN 102 3d_fullres 0 -tr nnUNetTrainer_M10_Lam05_Tr20 > "$BASE_DIR/logs_sweeps/gpu0_m10_lam05_tr20.log" 2>&1 &
PID0=$!
echo "✅ GPU 0 process launched (PID: $PID0) -> log: logs_sweeps/gpu0_m10_lam05_tr20.log"

# GPU 1: lambda = 1.50, Tr = 0.20
CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN 102 3d_fullres 0 -tr nnUNetTrainer_M10_Lam15_Tr20 > "$BASE_DIR/logs_sweeps/gpu1_m10_lam15_tr20.log" 2>&1 &
PID1=$!
echo "✅ GPU 1 process launched (PID: $PID1) -> log: logs_sweeps/gpu1_m10_lam15_tr20.log"

# GPU 2: lambda = 1.00, Tr = 0.10
CUDA_VISIBLE_DEVICES=2 $PYTHON_BIN 102 3d_fullres 0 -tr nnUNetTrainer_M10_Lam10_Tr10 > "$BASE_DIR/logs_sweeps/gpu2_m10_lam10_tr10.log" 2>&1 &
PID2=$!
echo "✅ GPU 2 process launched (PID: $PID2) -> log: logs_sweeps/gpu2_m10_lam10_tr10.log"

# GPU 3: lambda = 1.00, Tr = 0.30
CUDA_VISIBLE_DEVICES=3 $PYTHON_BIN 102 3d_fullres 0 -tr nnUNetTrainer_M10_Lam10_Tr30 > "$BASE_DIR/logs_sweeps/gpu3_m10_lam10_tr30.log" 2>&1 &
PID3=$!
echo "✅ GPU 3 process launched (PID: $PID3) -> log: logs_sweeps/gpu3_m10_lam10_tr30.log"

echo "=========================================================================="
echo "🎉 All 4 sweeps running in background across GPUs 0, 1, 2, 3!"
echo "PIDs: $PID0, $PID1, $PID2, $PID3"
echo "=========================================================================="

