"""
evaluate_nnunet.py
------------------
Evaluates the final nnU-Net v2 predictions against the ground truth test set.
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path

GT_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs")
PRED_DIR = Path("nnunet_data/nnUNet_results/predictions_test")

def compute_dice(pred, gt, class_idx):
    """Computes the Dice Similarity Coefficient for a specific class."""
    p = (pred == class_idx)
    g = (gt == class_idx)
    intersection = np.logical_and(p, g).sum()
    union = p.sum() + g.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union

def main():
    # Only grab valid NIfTI files
    test_files = [f for f in os.listdir(PRED_DIR) if f.endswith('.nii.gz')]
    
    if not test_files:
        print("Error: No predictions found in the target directory.")
        return
        
    print(f"Evaluating {len(test_files)} predicted cases against Ground Truth...\n")

    # Store scores for each class
    dice_scores = {1: [], 2: [], 3: []}

    for f in test_files:
        gt_path = GT_DIR / f
        pred_path = PRED_DIR / f
        
        if not gt_path.exists():
            print(f"Warning: Ground truth for {f} not found. Skipping.")
            continue

        gt_img = nib.load(gt_path).get_fdata()
        pred_img = nib.load(pred_path).get_fdata()

        for c in [1, 2, 3]:
            dice_scores[c].append(compute_dice(pred_img, gt_img, c))

    # Calculate means
    core_dice = np.mean(dice_scores[1])
    edema_dice = np.mean(dice_scores[2])
    enhancing_dice = np.mean(dice_scores[3])
    mean_dice = np.mean([core_dice, edema_dice, enhancing_dice])

    print("========= nnU-Net v2 TEST SET RESULTS =========")
    print(f"Core (1) Dice     : {core_dice:.4f}")
    print(f"Edema (2) Dice    : {edema_dice:.4f}")
    print(f"Enhancing (3) Dice: {enhancing_dice:.4f}")
    print(f"Overall Mean Dice : {mean_dice:.4f}")
    print("===============================================")

if __name__ == "__main__":
    main()