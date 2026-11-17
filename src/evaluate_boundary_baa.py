import os
import glob
import numpy as np
import nibabel as nib
import surface_distance as sd
import csv

gt_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/labelsTr"
pred_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset102_Restored/nnUNetTrainerSIAM_Boundary_BAA__nnUNetPlans__3d_fullres/fold_0/validation"
output_csv = "/mnt/DATA/EE23B009/EE23B009_BTP/brats_boundary_metrics_report_BAA.csv"

print("--- Starting Batch Evaluation for SIAM Boundary BAA ---")

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
    
    pred_arr = pred_nii.get_fdata().astype(np.uint8)
    gt_arr = gt_nii.get_fdata().astype(np.uint8)
    
    spacing = pred_nii.header.get_zooms()[:3]
    
    if np.any(pred_arr) and np.any(gt_arr):
        class_results = {}
        for c in [1, 2, 3]:
            pred_c = (pred_arr == c)
            gt_c = (gt_arr == c)
            if np.any(pred_c) and np.any(gt_c):
                surface_distances = sd.compute_surface_distances(gt_c, pred_c, spacing_mm=spacing)
                hd95 = sd.compute_robust_hausdorff(surface_distances, 95)
                assd_gt, assd_pred = sd.compute_average_surface_distance(surface_distances)
                assd = (assd_gt + assd_pred) / 2.0
                class_results[c] = {'hd95': hd95, 'assd': assd}
            else:
                class_results[c] = {'hd95': np.nan, 'assd': np.nan}
                
        # Overall
        pred_overall = (pred_arr > 0)
        gt_overall = (gt_arr > 0)
        if np.any(pred_overall) and np.any(gt_overall):
            surface_distances = sd.compute_surface_distances(gt_overall, pred_overall, spacing_mm=spacing)
            overall_hd95 = sd.compute_robust_hausdorff(surface_distances, 95)
            assd_gt, assd_pred = sd.compute_average_surface_distance(surface_distances)
            overall_assd = (assd_gt + assd_pred) / 2.0
        else:
            overall_hd95 = np.nan
            overall_assd = np.nan
            
        hd95_list.append({
            'overall': overall_hd95,
            '1': class_results[1]['hd95'],
            '2': class_results[2]['hd95'],
            '3': class_results[3]['hd95']
        })
        
        results.append({
            "Case": case_name,
            "HD95_Overall": round(overall_hd95, 2) if not np.isnan(overall_hd95) else "N/A",
            "HD95_C1": round(class_results[1]['hd95'], 2) if not np.isnan(class_results[1]['hd95']) else "N/A",
            "HD95_C2": round(class_results[2]['hd95'], 2) if not np.isnan(class_results[2]['hd95']) else "N/A",
            "HD95_C3": round(class_results[3]['hd95'], 2) if not np.isnan(class_results[3]['hd95']) else "N/A",
            "Status": "Success"
        })
        print(f"Processed: {case_name} | Overall HD95: {overall_hd95:.2f} | C1: {class_results[1]['hd95']:.2f} | C2: {class_results[2]['hd95']:.2f} | C3: {class_results[3]['hd95']:.2f}")
    else:
        results.append({
            "Case": case_name,
            "HD95_Overall": "N/A",
            "HD95_C1": "N/A",
            "HD95_C2": "N/A",
            "HD95_C3": "N/A",
            "Status": "Skipped (Empty Mask)"
        })
        print(f"Skipped: {case_name} (Empty mask)")

# Save to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Case", "HD95_Overall", "HD95_C1", "HD95_C2", "HD95_C3", "Status"])
    writer.writeheader()
    writer.writerows(results)

# Print overall summary statistics
if hd95_list:
    print("\n" + "="*40)
    print(" SUMMARY METRICS REPORT")
    print("="*40)
    
    overall_valid = [x['overall'] for x in hd95_list if not np.isnan(x['overall'])]
    c1_valid = [x['1'] for x in hd95_list if not np.isnan(x['1'])]
    c2_valid = [x['2'] for x in hd95_list if not np.isnan(x['2'])]
    c3_valid = [x['3'] for x in hd95_list if not np.isnan(x['3'])]
    
    print(f"Total Valid Cases Evaluated: {len(hd95_list)}")
    print(f"Median Overall HD95: {np.median(overall_valid):.2f} mm")
    print(f"Median Class 1 HD95: {np.median(c1_valid):.2f} mm")
    print(f"Median Class 2 HD95: {np.median(c2_valid):.2f} mm")
    print(f"Median Class 3 HD95: {np.median(c3_valid):.2f} mm")
    print(f"Report successfully saved to: {output_csv}")
    print("="*40)
