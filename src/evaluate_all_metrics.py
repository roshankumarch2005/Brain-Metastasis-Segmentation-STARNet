"""
evaluate_all_metrics.py
-----------------------
Evaluates both baseline nnU-Net and SIAM predictions.
Calculates both Dice Score (volume) and HD95 (boundaries).
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from medpy.metric.binary import hd95, dc

GT_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs")
PRED_BASELINE_DIR = Path("nnunet_data/nnUNet_results/predictions_test")
PRED_SIAM_DIR = Path("nnunet_data/nnUNet_results/predictions_test_siam")
PRED_SIAM_BAA_DIR = Path("nnunet_data/nnUNet_results/predictions_test_siam_baa")

def compute_metrics(pred_dir):
    test_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    
    # Dictionaries to hold scores
    dice_scores = {1: [], 2: [], 3: []}
    hd95_scores = {1: [], 2: [], 3: []}

    for f in test_files:
        gt_path = GT_DIR / f
        pred_path = pred_dir / f
        
        if not gt_path.exists(): continue

        gt_img = nib.load(gt_path).get_fdata()
        pred_img = nib.load(pred_path).get_fdata()

        for c in [1, 2, 3]:
            p = (pred_img == c)
            g = (gt_img == c)
            
            # Dice
            if p.sum() == 0 and g.sum() == 0:
                dice_scores[c].append(1.0)
            elif p.sum() == 0 or g.sum() == 0:
                dice_scores[c].append(0.0)
            else:
                dice_scores[c].append(dc(p, g))
            
            # HD95 (Requires both to have pixels, else it throws error)
            if p.sum() > 0 and g.sum() > 0:
                # voxelspacing=1.0 assumes isotropic for simplicity, usually you'd pass real spacing
                hd95_scores[c].append(hd95(p, g))

    print(f"\nResults for {pred_dir.name}:")
    for c in [1, 2, 3]:
        print(f"Class {c} | Dice: {np.mean(dice_scores[c]):.4f} | HD95: {np.mean(hd95_scores[c]):.4f} mm")
    print(f"OVERALL | Dice: {np.mean([np.mean(dice_scores[c]) for c in [1,2,3]]):.4f} | HD95: {np.mean([np.mean(hd95_scores[c]) for c in [1,2,3]]):.4f} mm")

if __name__ == "__main__":
    print("Evaluating Baseline nnU-Net...")
    compute_metrics(PRED_BASELINE_DIR)
    
    print("\n--------------------------------")
    print("Evaluating SIAM nnU-Net...")
    compute_metrics(PRED_SIAM_DIR)

    print("\n--------------------------------")
    print("Evaluating SIAM + BAA Loss nnU-Net...")
    if PRED_SIAM_BAA_DIR.exists():
        compute_metrics(PRED_SIAM_BAA_DIR)
    else:
        print("Predictions for SIAM_BAA not found yet!")

    PRED_SIAM_BAA_SENSITIVE_DIR = Path("nnunet_data/nnUNet_results/predictions_test_siam_baa_sensitive")
    print("\n--------------------------------")
    print("Evaluating SIAM + BAA Sensitive Loss nnU-Net...")
    if PRED_SIAM_BAA_SENSITIVE_DIR.exists():
        compute_metrics(PRED_SIAM_BAA_SENSITIVE_DIR)
    else:
        print("Predictions for SIAM_BAA_Sensitive not found yet!")

    PRED_ONLYDICE_DIR = Path("nnunet_data/nnUNet_results/predictions_test_onlydice")
    print("\n--------------------------------")
    print("Evaluating OnlyDice Ablation nnU-Net...")
    if PRED_ONLYDICE_DIR.exists():
        compute_metrics(PRED_ONLYDICE_DIR)
    else:
        print("Predictions for OnlyDice not found yet!")

    PRED_ONLYCE_DIR = Path("nnunet_data/nnUNet_results/predictions_test_onlyce")
    print("\n--------------------------------")
    print("Evaluating OnlyCE Ablation nnU-Net...")
    if PRED_ONLYCE_DIR.exists():
        compute_metrics(PRED_ONLYCE_DIR)
    else:
        print("Predictions for OnlyCE not found yet!")

    PRED_STAGE3_DIR = Path("nnunet_data/nnUNet_results/predictions_test_stage3")
    print("\n--------------------------------")
    print("Evaluating Truncated Stage 3 Architecture nnU-Net...")
    if PRED_STAGE3_DIR.exists():
        compute_metrics(PRED_STAGE3_DIR)
    else:
        print("Predictions for Stage 3 not found yet!")

