import os
import SimpleITK as sitk
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

def restore_mri(input_path, output_path):
    print(f"  -> Processing: {os.path.basename(input_path)}")
    
    # 1. Load Image
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    # 2. N4 Bias Field Correction
    mask = sitk.OtsuThreshold(image, 0, 1, 200) 
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    n4_image = corrector.Execute(image, mask)
    n4_array = sitk.GetArrayFromImage(n4_image)
    
    # 3. Non-Local Means Denoising
    sigma = estimate_sigma(n4_array, N=32)
    denoised_array = nlmeans(n4_array, 
                             sigma=sigma, 
                             mask=(n4_array > 0), 
                             patch_radius=1, 
                             block_radius=2, 
                             rician=True)
    
    # 4. Save the Restored Image
    denoised_image = sitk.GetImageFromArray(denoised_array)
    denoised_image.CopyInformation(n4_image)
    sitk.WriteImage(denoised_image, output_path)

def process_nnunet_test_set(input_dir, output_dir):
    # Automatically create the destination test folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all NIfTI files directly from the flat folder
    files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    print(f"Found {len(files)} files to process in {input_dir}")
    
    for filename in sorted(files):
        input_file = os.path.join(input_dir, filename)
        # Keep the exact same filename (e.g., 100108A_0000.nii.gz) for nnU-Net compatibility
        output_file = os.path.join(output_dir, filename)
        
        if not os.path.exists(output_file):
            restore_mri(input_file, output_file)
        else:
            print(f"  -> Skipping {filename} (Already exists)")

# Point directly to the raw BraTS test images and route them to Dataset102_Restored
raw_test_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/imagesTs"
restored_test_dir = "/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs"

if __name__ == "__main__":
    process_nnunet_test_set(raw_test_dir, restored_test_dir)