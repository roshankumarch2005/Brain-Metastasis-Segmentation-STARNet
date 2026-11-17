import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import label

INPUT_DIR = "nnunet_data/nnUNet_results/predictions_test"
OUTPUT_DIR = "nnunet_data/nnUNet_results/predictions_test_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum voxel thresholds (adjust as needed)
# 1 = Necrotic Core, 2 = Edema, 3 = Enhancing Tumor
MIN_COMPONENT_SIZES = {
    1: 15,   # Removes tiny isolated necrotic flecks
    2: 50,   # Edema should not exist as tiny disjoint specks
    3: 15    # Discards punctate vessel false-positives
}

pred_files = glob.glob(os.path.join(INPUT_DIR, "*.nii.gz"))
print(f"Found {len(pred_files)} prediction files to post-process.")

for fpath in pred_files:
    fname = os.path.basename(fpath)
    img = nib.load(fpath)
    data = img.get_fdata().astype(np.uint8)
    cleaned_data = np.zeros_like(data)

    for class_id, min_size in MIN_COMPONENT_SIZES.items():
        class_mask = (data == class_id)
        if not np.any(class_mask):
            continue
        
        # 26-connectivity in 3D (face, edge, and corner neighbors)
        structure = np.ones((3, 3, 3), dtype=bool)
        labeled_array, num_features = label(class_mask, structure=structure)
        
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled_array == comp_id)
            if np.sum(comp_mask) >= min_size:
                cleaned_data[comp_mask] = class_id

    # Preserve any labels not explicitly filtered
    for c in np.unique(data):
        if c not in MIN_COMPONENT_SIZES and c != 0:
            cleaned_data[data == c] = c

    out_img = nib.Nifti1Image(cleaned_data, affine=img.affine, header=img.header)
    nib.save(out_img, os.path.join(OUTPUT_DIR, fname))
    print(f"Processed: {fname}")

print(f"All cleaned masks saved to: {OUTPUT_DIR}")