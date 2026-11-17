"""
evaluate_saluja_test30.py
-------------------------
Calculates binary Dice score and HD95 (Hausdorff Distance 95th percentile)
on the 30 unseen test cases of Dataset105_UCSF_Saluja_Exact.
Uses true physical voxel spacing from NIfTI headers for accurate millimeter distances.
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from medpy.metric.binary import dc, hd95

GT_DIR_DGX = Path("/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/labelsTs")
GT_DIR_LOCAL = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/labelsTs")
GT_DIR = GT_DIR_DGX if GT_DIR_DGX.exists() else GT_DIR_LOCAL

def evaluate_predictions(pred_dir_path, csv_path=None):
    pred_dir = Path(pred_dir_path)
    if not pred_dir.exists():
        print(f"Error: Prediction directory {pred_dir} does not exist!")
        return

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
    if len(pred_files) == 0:
        print(f"Error: No .nii.gz prediction files found in {pred_dir}!")
        return

    print(f"==================================================================")
    print(f" Evaluating {len(pred_files)} Test Cases from: {pred_dir.name}")
    print(f" Ground Truth: {GT_DIR}")
    print(f"==================================================================")

    dice_list = []
    hd95_list = []
    detailed_results = []

    for f in pred_files:
        gt_path = GT_DIR / f
        pred_path = pred_dir / f

        if not gt_path.exists():
            print(f"Warning: Missing ground truth for {f}, skipping.")
            continue

        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)

        gt_data = (gt_nii.get_fdata() > 0).astype(np.uint8)
        pred_data = (pred_nii.get_fdata() > 0).astype(np.uint8)
        voxel_spacing = gt_nii.header.get_zooms()[:3]

        # 1. Compute Dice
        if gt_data.sum() == 0 and pred_data.sum() == 0:
            d = 1.0
        elif gt_data.sum() == 0 or pred_data.sum() == 0:
            d = 0.0
        else:
            d = dc(pred_data, gt_data)

        # 2. Compute HD95
        if gt_data.sum() > 0 and pred_data.sum() > 0:
            try:
                h = hd95(pred_data, gt_data, voxelspacing=voxel_spacing)
            except Exception as e:
                h = np.nan
        else:
            h = np.nan

        dice_list.append(d)
        if not np.isnan(h):
            hd95_list.append(h)

        case_id = f.replace('.nii.gz', '')
        detailed_results.append((case_id, d, h))

    mean_dice = np.mean(dice_list) * 100
    median_dice = np.median(dice_list) * 100
    mean_hd95 = np.mean(hd95_list)
    median_hd95 = np.median(hd95_list)

    print("\n---------------- Detailed Per-Case Results ----------------")
    print(f"{'Case ID':<25} | {'Dice (%)':<10} | {'HD95 (mm)':<10}")
    print("-" * 52)
    for cid, d, h in detailed_results:
        h_str = f"{h:.2f}" if not np.isnan(h) else "N/A"
        print(f"{cid:<25} | {d*100:>8.2f}% | {h_str:>8}")

    print("\n" + "=" * 62)
    print(f" SUMMARY METRICS (N = {len(dice_list)} test cases):")
    print(f" • Mean Dice Score:       {mean_dice:.2f}%")
    print(f" • Median Dice Score:     {median_dice:.2f}%")
    print(f" • Mean HD95 Distance:    {mean_hd95:.2f} mm")
    print(f" • Median HD95 Distance:  {median_hd95:.2f} mm")
    print("=" * 62)

    print("\n Comparison with Published Literature (Saluja et al. 2024 - Row L2):")
    print(f" • Saluja Reported Baseline: Enhancing Dice = 78.50% (Overall 80.00%)  |  Mean HD95 = 8.20 – 9.40 mm")
    print(f" • Our Model at This Step:   Dice = {mean_dice:.2f}%  |  Mean HD95 = {mean_hd95:.2f} mm (Median HD95 = {median_hd95:.2f} mm)")
    print("=" * 62 + "\n")

    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Case_ID", "Dice", "Dice_Percent", "HD95_mm"])
            for cid, d, h in detailed_results:
                writer.writerow([cid, f"{d:.6f}", f"{d*100:.2f}", f"{h:.4f}" if not np.isnan(h) else "NaN"])
            writer.writerow(["MEAN", f"{mean_dice/100.0:.6f}", f"{mean_dice:.2f}", f"{mean_hd95:.4f}"])
            writer.writerow(["MEDIAN", f"{median_dice/100.0:.6f}", f"{median_dice:.2f}", f"{median_hd95:.4f}"])
        print(f"💾 Saved full CSV metrics to: {csv_path}")

    return {
        "mean_dice": mean_dice,
        "median_dice": median_dice,
        "mean_hd95": mean_hd95,
        "median_hd95": median_hd95
    }

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_epoch750"
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    evaluate_predictions(target_dir, csv_file)
