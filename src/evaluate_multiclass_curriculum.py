#!/usr/bin/env python3
"""
evaluate_multiclass_curriculum.py
---------------------------------
Evaluates the 30 unseen test cases predicted by the Multi-Class Curriculum Model
(nnUNetTrainerSIAM_CurriculumBAA on Dataset102_Restored).

Calculates:
  - Class 1 (Necrotic Core) Dice
  - Class 2 (Peritumoral Edema) Dice
  - Class 3 (Enhancing Tumor / ET) Dice
  - Overall Mean & Median Dice
  - BraTS Composites: Whole Tumor (WT), Tumor Core (TC), Enhancing (ET)
  - 95th Percentile Hausdorff Distance (HD95 in mm) - Mean & Median
  - Head-to-head comparison against Model M5 (Standard BAA) and Literature
"""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from medpy.metric.binary import dc, hd95

GT_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs")
PRED_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/predictions_test_curriculum_baa")

def calc_hd95(p_mask, g_mask, spacing):
    if not np.any(p_mask) or not np.any(g_mask):
        return np.nan
    try:
        return hd95(p_mask, g_mask, voxelspacing=spacing)
    except Exception:
        return np.nan

def evaluate_all():
    print("==================================================================")
    print(" 🏥 EVALUATION: Multi-Class Curriculum Model (Dice -> BAA 1.5)")
    print("==================================================================")
    print(f"Ground Truth Folder : {GT_DIR}")
    print(f"Predictions Folder  : {PRED_DIR}")
    print("------------------------------------------------------------------")

    if not PRED_DIR.exists():
        print(f"Error: {PRED_DIR} does not exist!")
        return

    pred_files = sorted(list(PRED_DIR.glob("*.nii.gz")))
    if len(pred_files) == 0:
        print(f"Error: No prediction files in {PRED_DIR}!")
        return

    print(f"Found {len(pred_files)} test prediction cases.\n")

    d_c1, d_c2, d_c3, d_overall = [], [], [], []
    d_wt, d_tc, d_et = [], [], []
    hd95_wt_list = []
    case_rows = []

    print(f"{'Case ID':<15} | {'Core (C1)':<10} | {'Edema (C2)':<10} | {'ET (C3)':<10} | {'Mean Dice':<10} | {'WT HD95 (mm)':<12}")
    print("-" * 75)

    for pf in pred_files:
        cid = pf.name.replace(".nii.gz", "")
        gt_file = GT_DIR / pf.name
        if not gt_file.exists():
            continue

        gt_nii = nib.load(str(gt_file))
        pr_nii = nib.load(str(pf))
        gt_arr = gt_nii.get_fdata().astype(np.uint8)
        pr_arr = pr_nii.get_fdata().astype(np.uint8)
        spacing = pr_nii.header.get_zooms()[:3]

        # Multi-class Dice per compartment
        scores = []
        for c, target_list in [(1, d_c1), (2, d_c2), (3, d_c3)]:
            p = (pr_arr == c)
            g = (gt_arr == c)
            if p.sum() == 0 and g.sum() == 0:
                s = 1.0
            elif p.sum() == 0 or g.sum() == 0:
                s = 0.0
            else:
                s = dc(p, g)
            target_list.append(s)
            scores.append(s)

        c_mean = np.mean(scores)
        d_overall.append(c_mean)

        # BraTS Clinical Composites
        p_wt, g_wt = (pr_arr > 0), (gt_arr > 0)
        s_wt = 1.0 if (p_wt.sum()==0 and g_wt.sum()==0) else (0.0 if (p_wt.sum()==0 or g_wt.sum()==0) else dc(p_wt, g_wt))
        d_wt.append(s_wt)

        p_tc, g_tc = np.isin(pr_arr, [1, 3]), np.isin(gt_arr, [1, 3])
        s_tc = 1.0 if (p_tc.sum()==0 and g_tc.sum()==0) else (0.0 if (p_tc.sum()==0 or g_tc.sum()==0) else dc(p_tc, g_tc))
        d_tc.append(s_tc)

        d_et.append(scores[2])

        # HD95 on WT
        h = calc_hd95(p_wt, g_wt, spacing)
        if not np.isnan(h):
            hd95_wt_list.append(h)
            h_str = f"{h:.2f} mm"
        else:
            h_str = "N/A"

        case_rows.append((cid, scores[0]*100, scores[1]*100, scores[2]*100, c_mean*100, h))
        print(f"{cid:<15} | {scores[0]*100:>8.2f}% | {scores[1]*100:>8.2f}% | {scores[2]*100:>8.2f}% | {c_mean*100:>8.2f}% | {h_str:>12}")

    print("=" * 75)
    mean_c1, med_c1 = np.mean(d_c1)*100, np.median(d_c1)*100
    mean_c2, med_c2 = np.mean(d_c2)*100, np.median(d_c2)*100
    mean_c3, med_c3 = np.mean(d_c3)*100, np.median(d_c3)*100
    mean_ov, med_ov = np.mean(d_overall)*100, np.median(d_overall)*100
    mean_hd, med_hd = np.mean(hd95_wt_list), np.median(hd95_wt_list)

    print("\n📊 SUMMARY BENCHMARK METRICS (N = 30 Test Cases):")
    print(f" • Class 1 (Necrotic Core) Mean Dice : {mean_c1:.2f}% (Median: {med_c1:.2f}%)")
    print(f" • Class 2 (Edema) Mean Dice         : {mean_c2:.2f}% (Median: {med_c2:.2f}%)")
    print(f" • Class 3 (Enhancing) Mean Dice     : {mean_c3:.2f}% (Median: {med_c3:.2f}%)")
    print(f" • Overall Mean Dice                 : {mean_ov:.2f}%")
    print(f" • Overall Median Dice               : {med_ov:.2f}%")
    print(f" • Whole Tumor (WT) Mean Dice        : {np.mean(d_wt)*100:.2f}%")
    print(f" • Tumor Core (TC) Mean Dice         : {np.mean(d_tc)*100:.2f}%")
    print(f" • Enhancing Tumor (ET) Mean Dice    : {np.mean(d_et)*100:.2f}%")
    print(f" • Mean HD95 Boundary Distance       : {mean_hd:.2f} mm")
    print(f" • Median HD95 Boundary Distance     : {med_hd:.2f} mm")

    print("\n" + "=" * 75)
    print(" 🔬 COMPARISON: MULTI-CLASS MODELS (Restored Data)")
    print("=" * 75)
    print(f"{'Metric':<28} | {'Model M5 (Standard BAA)':<22} | {'Model M6 (Curriculum BAA)':<25}")
    print("-" * 75)
    print(f"{'Overall Mean Dice':<28} | {'79.71%':>22} | {mean_ov:>22.2f}%")
    print(f"{'Overall Median Dice':<28} | {'81.45%':>22} | {med_ov:>22.2f}%")
    print(f"{'Class 3 (ET) Mean Dice':<28} | {'79.58%':>22} | {mean_c3:>22.2f}%")
    print(f"{'Class 2 (Edema) Mean Dice':<28} | {'77.58%':>22} | {mean_c2:>22.2f}%")
    print(f"{'Mean HD95 (mm)':<28} | {'9.95 mm':>22} | {mean_hd:>19.2f} mm")
    print(f"{'Median HD95 (mm)':<28} | {'2.44 mm':>22} | {med_hd:>19.2f} mm")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    evaluate_all()

