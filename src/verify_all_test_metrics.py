import os
import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import surface_distance as sd
from medpy.metric.binary import dc

gt_dir = Path("nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs")

models = [
    ("M1: Baseline nnU-Net (Raw)", "nnunet_data/nnUNet_results/predictions_test"),
    ("M2: Baseline nnU-Net (Restored)", "nnunet_data/nnUNet_results/predictions_test_baseline_restored"),
    ("M3: SIAM nnU-Net", "nnunet_data/nnUNet_results/predictions_test_siam"),
    ("M4: SIAM + BAA (beta=0.75)", "nnunet_data/nnUNet_results/predictions_test_siam_baa"),
    ("M5: SIAM + BAA Sensitive (beta=1.5)", "nnunet_data/nnUNet_results/predictions_test_siam_baa_sensitive"),
    ("M6: SIAM + Boundary Alpha08", "nnunet_data/nnUNet_results/predictions_test_siam_boundary_alpha08"),
    ("M7: Ensemble (BAA + Alpha08)", "nnunet_data/nnUNet_results/predictions_test_ensemble_baa_alpha08"),
    ("A1: Ablation OnlyDice", "nnunet_data/nnUNet_results/predictions_test_onlydice"),
    ("A2: Ablation OnlyCE", "nnunet_data/nnUNet_results/predictions_test_onlyce"),
    ("A3: Truncated Stage 3", "nnunet_data/nnUNet_results/predictions_test_stage3")
]

print("=========================================================================================")
print(f"{'Model Name':<35} | {'Mean Dice':<10} | {'Med Dice':<10} | {'C3 Mean':<10} | {'C3 Med':<10} | {'Mean HD95':<10} | {'Med HD95':<10}")
print("=========================================================================================")

for name, p_path_str in models:
    p_path = Path(p_path_str)
    if not p_path.exists():
        print(f"Skipping {name}: Directory not found.")
        continue
        
    test_files = sorted([f for f in os.listdir(p_path) if f.endswith(".nii.gz")])
    if not test_files:
        print(f"Skipping {name}: No nii.gz files found.")
        continue

    dices_ov, dices_c1, dices_c2, dices_c3 = [], [], [], []
    hd95_list = []

    for f in test_files:
        gt_file = gt_dir / f
        if not gt_file.exists():
            continue
            
        gt_nii = nib.load(gt_file)
        pr_nii = nib.load(p_path / f)
        
        gt_arr = gt_nii.get_fdata()
        pr_arr = pr_nii.get_fdata()
        spacing = pr_nii.header.get_zooms()[:3]
        
        # Multiclass Dice
        case_d = []
        for c, d_list in [(1, dices_c1), (2, dices_c2), (3, dices_c3)]:
            p_c = (pr_arr == c)
            g_c = (gt_arr == c)
            if p_c.sum() == 0 and g_c.sum() == 0:
                val = 1.0
            elif p_c.sum() == 0 or g_c.sum() == 0:
                val = 0.0
            else:
                val = dc(p_c, g_c)
            d_list.append(val)
            case_d.append(val)
        dices_ov.append(np.mean(case_d))
        
        # Whole Tumor HD95 using surface_distance
        p_wt = (pr_arr > 0)
        g_wt = (gt_arr > 0)
        if np.any(p_wt) and np.any(g_wt):
            surfs = sd.compute_surface_distances(g_wt, p_wt, spacing_mm=spacing)
            h = sd.compute_robust_hausdorff(surfs, 95)
            hd95_list.append(h)
            
    m_dice = np.mean(dices_ov)
    med_dice = np.median(dices_ov)
    m_c3 = np.mean(dices_c3)
    med_c3 = np.median(dices_c3)
    
    m_hd95 = np.mean(hd95_list) if hd95_list else np.nan
    med_hd95 = np.median(hd95_list) if hd95_list else np.nan
    
    print(f"{name:<35} | {m_dice*100:>6.2f}%   | {med_dice*100:>6.2f}%   | {m_c3*100:>6.2f}%   | {med_c3*100:>6.2f}%   | {m_hd95:>6.2f} mm | {med_hd95:>6.2f} mm")

print("=========================================================================================")
