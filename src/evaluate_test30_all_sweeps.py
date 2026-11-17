#!/usr/bin/env python3
"""
evaluate_test30_all_sweeps.py
------------------------------------------------------
Evaluates the 30 unseen test cases for all 4 hyperparameter configurations of M10:
  1. Lam05_Tr20 (GPU 0)
  2. Lam15_Tr20 (GPU 1)
  3. Lam10_Tr10 (GPU 2)
  4. Lam10_Tr30 (GPU 3)

Ground truth: /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs
Calculates:
  - Multi-class Dice: C1 (Core), C2 (Edema), C3 (Enhancing)
  - BraTS Composites: WT, TC, ET
  - Overall Mean Dice = (C1 + C2 + C3) / 3.0
  - HD95 (mm): WT HD95, TC HD95, ET HD95
  - Mean and Median for all metrics
"""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from medpy.metric.binary import dc, hd95

GT_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs")
BASE_RES = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored")

MODELS = {
    "M10_Lam10_Tr10": BASE_RES / "preds_test30_m10_lam10_tr10",
    "M10_Lam05_Tr20": BASE_RES / "preds_test30_m10_lam05_tr20",
    "M10_Lam10_Tr30": BASE_RES / "preds_test30_m10_lam10_tr30",
    "M10_Lam15_Tr20": BASE_RES / "preds_test30_m10_lam15_tr20",
}

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

def evaluate_model(model_name, pred_dir):
    print(f"\n=======================================================")
    print(f"📊 Evaluating {model_name} on 30 Test Scans...")
    print(f"Prediction dir: {pred_dir}")
    print(f"=======================================================")
    
    pred_files = sorted([f for f in pred_dir.glob("*.nii.gz")])
    if len(pred_files) == 0:
        print(f"❌ No predictions found in {pred_dir}")
        return None

    records = []
    for p_path in pred_files:
        case_id = p_path.name.replace(".nii.gz", "")
        gt_path = GT_DIR / f"{case_id}.nii.gz"
        if not gt_path.exists():
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
    csv_file = Path(f"/mnt/DATA/EE23B009/EE23B009_BTP/{model_name}_test30_metrics.csv")
    df.to_csv(csv_file, index=False)
    
    summary = {
        "Model": model_name,
        "Overall_Mean_Dice": df["Dice_Mean"].mean(),
        "Overall_Median_Dice": df["Dice_Mean"].median(),
        "Core_C1_Mean": df["Dice_C1_Core"].mean(),
        "Edema_C2_Mean": df["Dice_C2_Edema"].mean(),
        "ET_C3_Mean": df["Dice_C3_ET"].mean(),
        "WT_Mean_Dice": df["Dice_WT"].mean(),
        "WT_Median_Dice": df["Dice_WT"].median(),
        "TC_Mean_Dice": df["Dice_TC"].mean(),
        "TC_Median_Dice": df["Dice_TC"].median(),
        "ET_Mean_Dice": df["Dice_ET"].mean(),
        "ET_Median_Dice": df["Dice_ET"].median(),
        "Mean_HD95_mm": df["HD95_WT_mm"].mean(),
        "Median_HD95_mm": df["HD95_WT_mm"].median(),
        "ET_Mean_HD95_mm": df["HD95_ET_mm"].mean(),
        "ET_Median_HD95_mm": df["HD95_ET_mm"].median(),
    }
    
    print(f"Results for {model_name}:")
    print(f"  • Overall Mean Dice : {summary['Overall_Mean_Dice']*100:.2f}% (Median: {summary['Overall_Median_Dice']*100:.2f}%)")
    print(f"  • Enhancing (ET)    : {summary['ET_Mean_Dice']*100:.2f}% (Median: {summary['ET_Median_Dice']*100:.2f}%)")
    print(f"  • Whole Tumor (WT)  : {summary['WT_Mean_Dice']*100:.2f}% (Median: {summary['WT_Median_Dice']*100:.2f}%)")
    print(f"  • Tumor Core (TC)   : {summary['TC_Mean_Dice']*100:.2f}% (Median: {summary['TC_Median_Dice']*100:.2f}%)")
    print(f"  • Mean HD95 (WT)    : {summary['Mean_HD95_mm']:.2f} mm (Median: {summary['Median_HD95_mm']:.2f} mm)")
    print(f"  • Mean HD95 (ET)    : {summary['ET_Mean_HD95_mm']:.2f} mm (Median: {summary['ET_Median_HD95_mm']:.2f} mm)")
    
    return summary

def main():
    all_summaries = []
    for model_name, pred_dir in MODELS.items():
        summary = evaluate_model(model_name, pred_dir)
        if summary is not None:
            all_summaries.append(summary)

    if len(all_summaries) > 0:
        sum_df = pd.DataFrame(all_summaries)
        sum_csv = Path("/mnt/DATA/EE23B009/EE23B009_BTP/m10_hyperparameter_sweep_test30_summary.csv")
        sum_df.to_csv(sum_csv, index=False)
        print("\n" + "=" * 100)
        print("🏆 FINAL COMPARISON: ALL 4 M10 HYPERPARAMETER CONFIGURATIONS (30 UNSEEN TEST SCANS)")
        print("=" * 100)
        print(f"{'Model / Hyperparams':<20} | {'Overall Dice':<14} | {'ET Dice':<14} | {'WT Dice':<14} | {'TC Dice':<14} | {'Mean HD95':<12} | {'Median HD95':<12}")
        print("-" * 100)
        for _, r in sum_df.iterrows():
            print(f"{r['Model']:<20} | {r['Overall_Mean_Dice']*100:6.2f}% ({r['Overall_Median_Dice']*100:5.2f}%) | {r['ET_Mean_Dice']*100:6.2f}% ({r['ET_Median_Dice']*100:5.2f}%) | {r['WT_Mean_Dice']*100:6.2f}% ({r['WT_Median_Dice']*100:5.2f}%) | {r['TC_Mean_Dice']*100:6.2f}% ({r['TC_Median_Dice']*100:5.2f}%) | {r['Mean_HD95_mm']:6.2f} mm   | {r['Median_HD95_mm']:6.2f} mm")
        print("=" * 100)

if __name__ == "__main__":
    main()

