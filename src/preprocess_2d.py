import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Paths
input_dir = "data/UCSF_BrainMetastases_TRAIN"
out_dir_images = "data/UCSF_2D_Slices/images"
out_dir_masks = "data/UCSF_2D_Slices/masks"

os.makedirs(out_dir_images, exist_ok=True)
os.makedirs(out_dir_masks, exist_ok=True)

def normalize(vol):
    """Min-Max normalization per volume."""
    min_val = np.min(vol)
    max_val = np.max(vol)
    return (vol - min_val) / (max_val - min_val + 1e-8)

# Get all subject folders
subjects = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for subj in tqdm(subjects, desc="Extracting 2D Slices"):
    subj_dir = os.path.join(input_dir, subj)
    
    # Locate the specific modalities and mask
    t1post_path = glob.glob(f"{subj_dir}/*_T1post.nii.gz")
    flair_path = glob.glob(f"{subj_dir}/*_FLAIR.nii.gz")
    t1pre_path = glob.glob(f"{subj_dir}/*_T1pre.nii.gz")
    
    # Matches either BraTS-seg.nii.gz or seg.nii.gz
    mask_path = glob.glob(f"{subj_dir}/*seg.nii.gz") 
    
    if not (t1post_path and flair_path and t1pre_path and mask_path):
        print(f"Skipping {subj}: Missing required modalities.")
        continue
        
    # Load 3D volumes into numpy arrays
    t1post_vol = nib.load(t1post_path[0]).get_fdata()
    flair_vol = nib.load(flair_path[0]).get_fdata()
    t1pre_vol = nib.load(t1pre_path[0]).get_fdata()
    mask_vol = nib.load(mask_path[0]).get_fdata()
    
    # Normalize image volumes to [0, 1] range for the neural network
    t1post_vol = normalize(t1post_vol)
    flair_vol = normalize(flair_vol)
    t1pre_vol = normalize(t1pre_vol)
    
    # Extract slices along the axial plane (z-axis, index 2)
    num_slices = mask_vol.shape[2]
    
    for i in range(num_slices):
        mask_slice = mask_vol[:, :, i]
        
        # Filter: Only save slices that contain actual tumor tissue.
        # This prevents the model from being overwhelmed by empty background slices.
        if np.sum(mask_slice) > 0:
            
            # Stack modalities into a (3, H, W) tensor for SMP's in_channels=3
            img_slice = np.stack([
                t1post_vol[:, :, i], 
                flair_vol[:, :, i], 
                t1pre_vol[:, :, i]
            ], axis=0).astype(np.float32)
            
            mask_slice = np.expand_dims(mask_slice, axis=0).astype(np.float32)
            
            # Save format: SubjectID_SliceNumber.npy
            base_name = f"{subj}_slice_{i:03d}"
            np.save(os.path.join(out_dir_images, base_name + ".npy"), img_slice)
            np.save(os.path.join(out_dir_masks, base_name + ".npy"), mask_slice)
