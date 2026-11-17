#!/usr/bin/env python3
"""
evaluate_decoder1_halfres_multiclass_test30.py
---------------------------------------------
Evaluates the 30 unseen test cases predicted by Method 2:
Decoder-1 Half-Resolution Bypass Multiclass nnUNet (Dataset102_Restored).

Calculates:
  - Multi-class Dice: Necrotic Core (C1), Edema (C2), Enhancing Tumor (C3)
  - BraTS Composites: Whole Tumor (WT), Tumor Core (TC), Enhancing (ET)
  - Physical 95th Percentile Hausdorff Distance (HD95 in mm) for WT, TC, and ET
  - Saves metrics to CSV and detailed summary table
"""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from medpy.metric.binary import dc, hd95

GT_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs")

def calc_hd95(p_mask, g_mask, spacing):
    if not np.any(p_mask) or not np.any(g_mask):
        return np.nan
    try:
        return float(hd95(p_mask, g_mask, voxelspacing=spacing))
    except Exception:
        return np.nan

def calc_dice(p_mask, g_mask):
    p_sum = p_mask.sum()
    g_sum = g_mask.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    return float(dc(p_mask, g_mask))

def main():
    pred_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_decoder1_halfres_multiclass_final")
    
    if not pred_dir.exists():
        print(f"Error: {pred_dir} does not exist!")
        sys.exit(1)

    pred_files = sorted([f for f in pred_dir.glob("*.nii.gz")])
    if len(pred_files) == 0:
        print(f"Error: No prediction .nii.gz files found in {pred_dir}!")
        sys.exit(1)

    print("=" * 80)
    print(" 🏥 METHOD 2: DECODER-1 HALF-RES BYPASS MULTICLASS BENCHMARK")
    print(f" Predictions Dir: {pred_dir}")
    print(f" Ground Truth   : {GT_DIR}")
    print(f" Cases Found    : {len(pred_files)}")
    print("=" * 80)

    rows = []

    for pf in pred_files:
        cid = pf.name.replace(".nii.gz", "")
        gt_file = GT_DIR / pf.name
        if not gt_file.exists():
            print(f"Warning: {gt_file} not found, skipping {cid}")
            continue

        gt_nii = nib.load(str(gt_file))
        pr_nii = nib.load(str(pf))
        gt_arr = gt_nii.get_fdata().astype(np.uint8)
        pr_arr = pr_nii.get_fdata().astype(np.uint8)
        spacing = pr_nii.header.get_zooms()[:3]

        # Individual subregions
        dice_c1 = calc_dice(pr_arr == 1, gt_arr == 1) # Necrotic Core
        dice_c2 = calc_dice(pr_arr == 2, gt_arr == 2) # Edema
        dice_c3 = calc_dice(pr_arr == 3, gt_arr == 3) # Enhancing Tumor
        dice_mean = (dice_c1 + dice_c2 + dice_c3) / 3.0

        # Clinical Composites
        p_wt, g_wt = (pr_arr > 0), (gt_arr > 0)
        dice_wt = calc_dice(p_wt, g_wt)
        hd95_wt = calc_hd95(p_wt, g_wt, spacing)

        p_tc, g_tc = np.isin(pr_arr, [1, 3]), np.isin(gt_arr, [1, 3])
        dice_tc = calc_dice(p_tc, g_tc)
        hd95_tc = calc_hd95(p_tc, g_tc, spacing)

        p_et, g_et = (pr_arr == 3), (gt_arr == 3)
        dice_et = dice_c3
        hd95_et = calc_hd95(p_et, g_et, spacing)

        rows.append({
            "Case_ID": cid,
            "Dice_C1_Core": dice_c1,
            "Dice_C2_Edema": dice_c2,
            "Dice_C3_ET": dice_c3,
            "Dice_Mean": dice_mean,
            "Dice_WT": dice_wt,
            "Dice_TC": dice_tc,
            "Dice_ET": dice_et,
            "HD95_WT_mm": hd95_wt,
            "HD95_TC_mm": hd95_tc,
            "HD95_ET_mm": hd95_et
        })

    df = pd.DataFrame(rows)
    csv_out = Path("/mnt/DATA/EE23B009/EE23B009_BTP/method2_decoder1_halfres_multiclass_test30_metrics.csv")
    df.to_csv(csv_out, index=False)

    print("\n" + "=" * 90)
    print(f"{'Case ID':<12} | {'Core(C1)':<9} | {'Edema(C2)':<9} | {'ET(C3)':<9} | {'Mean Dice':<9} | {'WT Dice':<9} | {'WT HD95(mm)':<11}")
    print("-" * 90)
    for _, r in df.iterrows():
        wt_hd_str = f"{r['HD95_WT_mm']:.2f}" if not np.isnan(r['HD95_WT_mm']) else "N/A"
        print(f"{r['Case_ID']:<12} | {r['Dice_C1_Core']*100:6.2f}%  | {r['Dice_C2_Edema']*100:6.2f}%  | {r['Dice_C3_ET']*100:6.2f}%  | {r['Dice_Mean']*100:6.2f}%  | {r['Dice_WT']*100:6.2f}%  | {wt_hd_str:>8} mm")

    print("=" * 90)
    print("\n📊 AGGREGATE TEST SET RESULTS (30 Unseen BraTS Scans):")
    print(f"  • Mean Subregion Dice  : {df['Dice_Mean'].mean()*100:.2f}% (Median: {df['Dice_Mean'].median()*100:.2f}%)")
    print(f"  • Enhancing Tumor (ET) : {df['Dice_ET'].mean()*100:.2f}% (Median: {df['Dice_ET'].median()*100:.2f}%)")
    print(f"  • Peritumoral Edema    : {df['Dice_C2_Edema'].mean()*100:.2f}% (Median: {df['Dice_C2_Edema'].median()*100:.2f}%)")
    print(f"  • Necrotic Core        : {df['Dice_C1_Core'].mean()*100:.2f}% (Median: {df['Dice_C1_Core'].median()*100:.2f}%)")
    print(f"  • Whole Tumor (WT) Dice: {df['Dice_WT'].mean()*100:.2f}% (Median: {df['Dice_WT'].median()*100:.2f}%)")
    print(f"  • Tumor Core (TC) Dice : {df['Dice_TC'].mean()*100:.2f}% (Median: {df['Dice_TC'].median()*100:.2f}%)")
    print(f"  • Whole Tumor HD95     : {df['HD95_WT_mm'].mean():.2f} mm (Median: {df['HD95_WT_mm'].median():.2f} mm)")
    print(f"  • Enhancing Tumor HD95 : {df['HD95_ET_mm'].mean():.2f} mm (Median: {df['HD95_ET_mm'].median():.2f} mm)")
    print(f"  • Detailed CSV saved   : {csv_out}")
    print("=" * 90)

if __name__ == "__main__":
    main()

