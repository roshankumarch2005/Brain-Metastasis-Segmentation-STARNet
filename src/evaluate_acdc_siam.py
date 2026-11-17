import os
import json
import glob
import numpy as np
import nibabel as nib
from medpy.metric.binary import hd95, assd

gt_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset027_ACDC/labelsTr"
pred_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset027_ACDC/nnUNetTrainerSIAM__nnUNetPlans__3d_fullres/fold_0/validation"
output_json = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset027_ACDC/nnUNetTrainerSIAM__nnUNetPlans__3d_fullres/fold_0/validation/extended_metrics_acdc.json"

pred_files = sorted(glob.glob(os.path.join(pred_dir, "patient*.nii.gz")))

classes = {
    1: "RV",
    2: "MYO",
    3: "LV"
}

results_per_case = []
metrics_summary = {
    c: {"dice": [], "iou": [], "hd95": [], "assd": []} for c in classes
}

print(f"Found {len(pred_files)} validation cases in {pred_dir}")

for p_path in pred_files:
    case_name = os.path.basename(p_path)
    gt_path = os.path.join(gt_dir, case_name)
    if not os.path.exists(gt_path):
        print(f"Warning: GT not found for {case_name}")
        continue
    
    p_img = nib.load(p_path)
    g_img = nib.load(gt_path)
    
    p_data = p_img.get_fdata().astype(np.uint8)
    g_data = g_img.get_fdata().astype(np.uint8)
    voxel_spacing = p_img.header.get_zooms()[:3]
    
    case_res = {"case": case_name, "metrics": {}}
    
    for c_id, c_name in classes.items():
        p_mask = (p_data == c_id)
        g_mask = (g_data == c_id)
        
        tp = np.logical_and(p_mask, g_mask).sum()
        fp = np.logical_and(p_mask, ~g_mask).sum()
        fn = np.logical_and(~p_mask, g_mask).sum()
        
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 1.0
        iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 1.0
        
        # HD95 and ASSD
        if p_mask.sum() > 0 and g_mask.sum() > 0:
            try:
                h = float(hd95(p_mask, g_mask, voxelspacing=voxel_spacing))
            except Exception:
                h = np.nan
            try:
                a = float(assd(p_mask, g_mask, voxelspacing=voxel_spacing))
            except Exception:
                a = np.nan
        else:
            h = np.nan
            a = np.nan
            
        case_res["metrics"][c_name] = {
            "dice": float(dice),
            "iou": float(iou),
            "hd95_mm": h,
            "assd_mm": a
        }
        
        metrics_summary[c_id]["dice"].append(dice)
        metrics_summary[c_id]["iou"].append(iou)
        if not np.isnan(h):
            metrics_summary[c_id]["hd95"].append(h)
        if not np.isnan(a):
            metrics_summary[c_id]["assd"].append(a)
            
    results_per_case.append(case_res)

summary = {}
for c_id, c_name in classes.items():
    d_list = metrics_summary[c_id]["dice"]
    i_list = metrics_summary[c_id]["iou"]
    h_list = metrics_summary[c_id]["hd95"]
    a_list = metrics_summary[c_id]["assd"]
    summary[c_name] = {
        "mean_dice": float(np.mean(d_list)),
        "median_dice": float(np.median(d_list)),
        "mean_iou": float(np.mean(i_list)),
        "mean_hd95_mm": float(np.mean(h_list)),
        "median_hd95_mm": float(np.median(h_list)),
        "mean_assd_mm": float(np.mean(a_list)),
        "median_assd_mm": float(np.median(a_list))
    }

overall_dice = np.mean([summary[c]["mean_dice"] for c in classes.values()])
overall_hd95_median = np.median([summary[c]["median_hd95_mm"] for c in classes.values()])

final_out = {
    "dataset": "Dataset027_ACDC",
    "model": "SIAM_nnU-Net",
    "num_validation_cases": len(results_per_case),
    "classes": classes,
    "class_summary": summary,
    "overall_mean_dice": float(overall_dice),
    "overall_median_hd95_mm": float(overall_hd95_median),
    "per_case": results_per_case
}

with open(output_json, "w") as f:
    json.dump(final_out, f, indent=4)

print("\n=== Dataset027_ACDC SIAM nnU-Net Evaluation Complete ===")
print(f"Overall Mean Dice: {overall_dice:.4f}")
for c_name, vals in summary.items():
    print(f"  {c_name}: Dice = {vals['mean_dice']:.4f} (Med {vals['median_dice']:.4f}) | HD95 = {vals['median_hd95_mm']:.2f} mm | ASSD = {vals['median_assd_mm']:.2f} mm")
