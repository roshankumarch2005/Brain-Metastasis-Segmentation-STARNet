import os
import numpy as np
import nibabel as nib
from pathlib import Path
from medpy.metric.binary import sensitivity, precision

GT_DIR = Path("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs")
PRED_DIRS = {
    "Baseline nnU-Net": Path("nnunet_data/nnUNet_results/predictions_test"),
    "SIAM nnU-Net": Path("nnunet_data/nnUNet_results/predictions_test_siam"),
    "SIAM+BAA (beta=0.75)": Path("nnunet_data/nnUNet_results/predictions_test_siam_baa"),
    "SIAM+BAA Sensitive (beta=1.5)": Path("nnunet_data/nnUNet_results/predictions_test_siam_baa_sensitive"),
}

def compute_sens_prec(pred_dir, model_name):
    if not pred_dir.exists():
        return
        
    test_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]
    
    sens_scores = {1: [], 2: [], 3: []}
    prec_scores = {1: [], 2: [], 3: []}

    for f in test_files:
        gt_path = GT_DIR / f
        pred_path = pred_dir / f
        
        if not gt_path.exists(): continue

        gt_img = nib.load(gt_path).get_fdata()
        pred_img = nib.load(pred_path).get_fdata()

        for c in [1, 2, 3]:
            p = (pred_img == c)
            g = (gt_img == c)
            
            # Sensitivity (Recall)
            if g.sum() == 0 and p.sum() == 0:
                sens_scores[c].append(1.0)
                prec_scores[c].append(1.0)
            elif g.sum() == 0 and p.sum() > 0:
                sens_scores[c].append(1.0)
                prec_scores[c].append(0.0)
            elif g.sum() > 0 and p.sum() == 0:
                sens_scores[c].append(0.0)
                prec_scores[c].append(1.0)
            else:
                sens_scores[c].append(sensitivity(p, g))
                prec_scores[c].append(precision(p, g))

    print(f"\n{model_name}:")
    for c in [1, 2, 3]:
        print(f"  Class {c} | Sensitivity (Recall): {np.mean(sens_scores[c]):.4f} | Precision: {np.mean(prec_scores[c]):.4f}")
    
    mean_sens = np.mean([np.mean(sens_scores[c]) for c in [1,2,3]])
    mean_prec = np.mean([np.mean(prec_scores[c]) for c in [1,2,3]])
    print(f"  OVERALL | Sensitivity (Recall): {mean_sens:.4f} | Precision: {mean_prec:.4f}")

if __name__ == "__main__":
    print("Calculating Sensitivity and Precision for all models...")
    for name, directory in PRED_DIRS.items():
        compute_sens_prec(directory, name)
