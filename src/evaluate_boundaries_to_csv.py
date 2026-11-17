import os
import glob
import numpy as np
import nibabel as nib
import surface_distance as sd
import csv

gt_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Dataset102_Restored/labelsTr"
pred_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/predictions_test"
output_csv = "brats_boundary_metrics_report.csv"

print("--- Starting Batch Evaluation & CSV Report Generation ---")

results = []
hd95_list = []
assd_list = []

for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
    case_name = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, case_name)
    
    if not os.path.exists(gt_path):
        print(f"Skipping {case_name}: Ground truth file not found.")
        continue
        
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)
    
    pred_arr = pred_nii.get_fdata().astype(np.bool_)
    gt_arr = gt_nii.get_fdata().astype(np.bool_)
    
    spacing = pred_nii.header.get_zooms()[:3]
    
    if np.any(pred_arr) and np.any(gt_arr):
        surface_distances = sd.compute_surface_distances(gt_arr, pred_arr, spacing_mm=spacing)
        
        # Compute metrics
        hd95 = sd.compute_robust_hausdorff(surface_distances, 95)
        assd_gt, assd_pred = sd.compute_average_surface_distance(surface_distances)
        assd = (assd_gt + assd_pred) / 2.0
        
        hd95_list.append(hd95)
        assd_list.append(assd)
        
        results.append({
            "Case": case_name,
            "HD95_mm": round(hd95, 2),
            "ASSD_mm": round(assd, 2),
            "Status": "Success"
        })
        print(f"Processed: {case_name} | HD95: {hd95:.2f} mm | ASSD: {assd:.2f} mm")
    else:
        results.append({
            "Case": case_name,
            "HD95_mm": "N/A",
            "ASSD_mm": "N/A",
            "Status": "Skipped (Empty Mask)"
        })
        print(f"Skipped: {case_name} (Empty mask)")

# Save to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Case", "HD95_mm", "ASSD_mm", "Status"])
    writer.writeheader()
    writer.writerows(results)

# Print overall summary statistics
if hd95_list:
    print("\n" + "="*40)
    print(" SUMMARY METRICS REPORT")
    print("="*40)
    print(f"Total Valid Cases Evaluated: {len(hd95_list)}")
    print(f"Mean HD95: {np.mean(hd95_list):.2f} mm (± {np.std(hd95_list):.2f})")
    print(f"Median HD95: {np.median(hd95_list):.2f} mm")
    print(f"Mean ASSD: {np.mean(assd_list):.2f} mm (± {np.std(assd_list):.2f})")
    print(f"Median ASSD: {np.median(assd_list):.2f} mm")
    print(f"Report successfully saved to: {output_csv}")
    print("="*40)
    