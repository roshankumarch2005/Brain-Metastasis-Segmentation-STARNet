#!/usr/bin/env python3
"""
evaluate_method3_siam_curriculum_multiclass_test30.py
------------------------------------------------------
Evaluates the 30 unseen test cases predicted by Method 3 + SIAM + Curriculum BAA
(Dataset102_Restored).

Calculates:
  - Multi-class Dice: Necrotic Core (C1), Edema (C2), Enhancing Tumor (C3)
  - BraTS Composites: Whole Tumor (WT), Tumor Core (TC), Enhancing (ET)
  - Physical 95th Percentile Hausdorff Distance (HD95 in mm) for WT, TC, and ET
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
    pred_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_method3_siam_curriculum_multiclass_final")
    
    if not pred_dir.exists():
        print(f"Error: {pred_dir} does not exist!")
        sys.exit(1)

    pred_files = sorted([f for f in pred_dir.glob("*.nii.gz")])
    if len(pred_files) == 0:
        print(f"Error: No .nii.gz files found in {pred_dir}")
        sys.exit(1)

    records = []
    print(f"\nEvaluating {len(pred_files)} predictions from: {pred_dir}")
    print(f"Ground truth source: {GT_DIR}\n")

    for p_path in pred_files:
        case_id = p_path.name.replace(".nii.gz", "")
        gt_path = GT_DIR / f"{case_id}.nii.gz"
        
        if not gt_path.exists():
            print(f"Warning: Ground truth missing for {case_id}, skipping...")
            continue

        p_nii = nib.load(str(p_path))
        g_nii = nib.load(str(gt_path))

        p_data = np.asanyarray(p_nii.dataobj, dtype=np.uint8)
        g_data = np.asanyarray(g_nii.dataobj, dtype=np.uint8)
        spacing = tuple(p_nii.header.get_zooms()[:3])

        # Subregions
        c1_p, c1_g = (p_data == 1), (g_data == 1) # Necrotic Core
        c2_p, c2_g = (p_data == 2), (g_data == 2) # Edema
        c3_p, c3_g = (p_data == 3), (g_data == 3) # Enhancing

        # BraTS Composites
        wt_p, wt_g = (p_data > 0), (g_data > 0)
        tc_p, tc_g = np.isin(p_data, [1, 3]), np.isin(g_data, [1, 3])
        et_p, et_g = c3_p, c3_g

        d_c1 = calc_dice(c1_p, c1_g)
        d_c2 = calc_dice(c2_p, c2_g)
        d_c3 = calc_dice(c3_p, c3_g)
        d_wt = calc_dice(wt_p, wt_g)
        d_tc = calc_dice(tc_p, tc_g)
        d_et = calc_dice(et_p, et_g)
        d_mean = (d_c1 + d_c2 + d_c3) / 3.0

        hd_wt = calc_hd95(wt_p, wt_g, spacing)
        hd_tc = calc_hd95(tc_p, tc_g, spacing)
        hd_et = calc_hd95(et_p, et_g, spacing)

        records.append({
            "Case_ID": case_id,
            "Dice_C1_Core": d_c1,
            "Dice_C2_Edema": d_c2,
            "Dice_C3_ET": d_c3,
            "Dice_Mean": d_mean,
            "Dice_WT": d_wt,
            "Dice_TC": d_tc,
            "Dice_ET": d_et,
            "HD95_WT_mm": hd_wt,
            "HD95_TC_mm": hd_tc,
            "HD95_ET_mm": hd_et,
        })

    df = pd.DataFrame(records)
    csv_out = Path("/mnt/DATA/EE23B009/EE23B009_BTP/method3_siam_curriculum_multiclass_test30_metrics.csv")
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

