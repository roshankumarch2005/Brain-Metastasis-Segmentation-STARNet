#!/usr/bin/env python3
"""
evaluate_m5_raw.py
------------------
Evaluates M5 RAW (SIAM + BAA Sensitive beta=1.5 on RAW multi-class MRI).
Computes:
  - Class 1 (Core), Class 2 (Edema), Class 3 (Enhancing) Dice
  - Composite BraTS regions: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
  - Mean & Median HD95 (Boundary error in millimeters)
  - Direct Head-to-Head Comparison against M5 Restored (N4+NLM)
"""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from medpy.metric.binary import dc

# Surface distance for HD95 (fallback to medpy hd95 if not installed)
try:
    import surface_distance as sd
    HAS_SURF_DIST = True
except ImportError:
    from medpy.metric.binary import hd95
    HAS_SURF_DIST = False

def find_gt_dir():
    candidates = [
        Path("nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs"),
        Path("nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs"),
        Path("/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs"),
        Path("/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs"),
        Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs"),
        Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs"),
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*.nii.gz"))) > 0:
            return c
    raise FileNotFoundError("Could not locate ground truth labelsTs directory!")

def find_pred_dir(custom_path=None):
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    candidates = [
        Path("nnunet_data/nnUNet_results/predictions_test_m5_raw"),
        Path("nnunet_data/nnUNet_results/predictions_test_m5_raw_baa_sensitive"),
        Path("/Data3/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/predictions_test_m5_raw"),
        Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/predictions_test_m5_raw"),
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*.nii.gz"))) > 0:
            return c
    return candidates[0]

def calc_hd95(p_mask, g_mask, spacing):
    if not np.any(p_mask) or not np.any(g_mask):
        return np.nan
    try:
        if HAS_SURF_DIST:
            surfs = sd.compute_surface_distances(g_mask, p_mask, spacing_mm=spacing)
            return sd.compute_robust_hausdorff(surfs, 95)
        else:
            return hd95(p_mask, g_mask, voxelspacing=spacing)
    except Exception:
        return np.nan

def main():
    pred_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    gt_dir = find_gt_dir()
    pred_dir = find_pred_dir(pred_path_arg)

    print("==================================================================")
    print(" 🏥 EVALUATION: M5 RAW (SIAM + BAA Sensitive beta=1.5 on RAW MRI)")
    print("==================================================================")
    print(f"Ground Truth Folder : {gt_dir}")
    print(f"Predictions Folder  : {pred_dir}")
    print("------------------------------------------------------------------")

    if not pred_dir.exists() or len(list(pred_dir.glob("*.nii.gz"))) == 0:
        print(f"[!] Predictions directory {pred_dir} is empty or not found.")
        print(f"    Please run the nnUNetv2_predict command first!")
        return

    pred_files = sorted(list(pred_dir.glob("*.nii.gz")))
    print(f"Found {len(pred_files)} test prediction cases.\n")

    d_c1, d_c2, d_c3, d_overall = [], [], [], []
    d_wt, d_tc, d_et = [], [], []
    hd95_wt_list = []

    print(f"{'Case ID':<15} | {'Core (C1)':<10} | {'Edema (C2)':<10} | {'ET (C3)':<10} | {'Mean Dice':<10} | {'WT HD95 (mm)':<12}")
    print("-" * 75)

    for pf in pred_files:
        cid = pf.name.replace(".nii.gz", "")
        gt_file = gt_dir / pf.name
        if not gt_file.exists():
            continue

        gt_nii = nib.load(gt_file)
        pr_nii = nib.load(pf)
        gt_arr = gt_nii.get_fdata().astype(np.uint8)
        pr_arr = pr_nii.get_fdata().astype(np.uint8)
        spacing = pr_nii.header.get_zooms()[:3]

        # Multi-class Dice
        case_scores = []
        for c, target_list in [(1, d_c1), (2, d_c2), (3, d_c3)]:
            p = (pr_arr == c)
            g = (gt_arr == c)
            if p.sum() == 0 and g.sum() == 0:
                score = 1.0
            elif p.sum() == 0 or g.sum() == 0:
                score = 0.0
            else:
                score = dc(p, g)
            target_list.append(score)
            case_scores.append(score)
        
        c_mean = np.mean(case_scores)
        d_overall.append(c_mean)

        # BraTS Clinical Composites
        # WT: 1 + 2 + 3
        p_wt, g_wt = (pr_arr > 0), (gt_arr > 0)
        d_wt.append(1.0 if (p_wt.sum()==0 and g_wt.sum()==0) else (0.0 if (p_wt.sum()==0 or g_wt.sum()==0) else dc(p_wt, g_wt)))
        
        # TC: 1 + 3
        p_tc, g_tc = np.isin(pr_arr, [1, 3]), np.isin(gt_arr, [1, 3])
        d_tc.append(1.0 if (p_tc.sum()==0 and g_tc.sum()==0) else (0.0 if (p_tc.sum()==0 or g_tc.sum()==0) else dc(p_tc, g_tc)))
        
        # ET: 3
        d_et.append(case_scores[2])

        # HD95
        h = calc_hd95(p_wt, g_wt, spacing)
        if not np.isnan(h):
            hd95_wt_list.append(h)
            h_str = f"{h:.2f} mm"
        else:
            h_str = "N/A"

        print(f"{cid:<15} | {case_scores[0]*100:>8.2f}% | {case_scores[1]*100:>8.2f}% | {case_scores[2]*100:>8.2f}% | {c_mean*100:>8.2f}% | {h_str:>12}")

    print("=" * 75)
    print("\n📊 SUMMARY BENCHMARK METRICS (N = 30 Test Cases):")
    print(f" • Class 1 (Necrotic Core) Mean Dice : {np.mean(d_c1)*100:.2f}% (Median: {np.median(d_c1)*100:.2f}%)")
    print(f" • Class 2 (Edema) Mean Dice         : {np.mean(d_c2)*100:.2f}% (Median: {np.median(d_c2)*100:.2f}%)")
    print(f" • Class 3 (Enhancing) Mean Dice     : {np.mean(d_c3)*100:.2f}% (Median: {np.median(d_c3)*100:.2f}%)")
    print(f" • Overall Mean Dice                 : {np.mean(d_overall)*100:.2f}%")
    print(f" • Overall Median Dice               : {np.median(d_overall)*100:.2f}%")
    print(f" • Whole Tumor (WT) Mean Dice        : {np.mean(d_wt)*100:.2f}%")
    print(f" • Tumor Core (TC) Mean Dice         : {np.mean(d_tc)*100:.2f}%")
    print(f" • Enhancing Tumor (ET) Mean Dice    : {np.mean(d_et)*100:.2f}%")
    print(f" • Mean HD95 Boundary Distance       : {np.mean(hd95_wt_list):.2f} mm")
    print(f" • Median HD95 Boundary Distance     : {np.median(hd95_wt_list):.2f} mm")

    print("\n" + "=" * 75)
    print(" 🔬 HEAD-TO-HEAD: M5 RAW vs. M5 RESTORED (Impact of N4+NLM)")
    print("=" * 75)
    print(f"{'Metric':<28} | {'M5 RAW (Our New Run)':<22} | {'M5 RESTORED (Previous)':<22}")
    print("-" * 75)
    print(f"{'Overall Mean Dice':<28} | {np.mean(d_overall)*100:>19.2f}% | {'79.71%':>22}")
    print(f"{'Overall Median Dice':<28} | {np.median(d_overall)*100:>19.2f}% | {'81.45%':>22}")
    print(f"{'Class 3 (ET) Mean Dice':<28} | {np.mean(d_c3)*100:>19.2f}% | {'79.58%':>22}")
    print(f"{'Class 2 (Edema) Mean Dice':<28} | {np.mean(d_c2)*100:>19.2f}% | {'77.58%':>22}")
    print(f"{'Mean HD95 (mm)':<28} | {np.mean(hd95_wt_list):>19.2f} mm | {'9.95 mm':>22}")
    print(f"{'Median HD95 (mm)':<28} | {np.median(hd95_wt_list):>19.2f} mm | {'2.44 mm':>22}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
