import SimpleITK as sitk
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

def restore_mri(input_path, output_path):
    print(f"Processing: {input_path}")
    
    # 1. Load Image
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    # 2. N4 Bias Field Correction
    print("Running N4 Bias Field Correction...")
    # Generate a basic mask to tell the algorithm where the brain is
    mask = sitk.OtsuThreshold(image, 0, 1, 200) 
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    n4_image = corrector.Execute(image, mask)
    
    # Convert to numpy array for the DIPY denoising algorithm
    n4_array = sitk.GetArrayFromImage(n4_image)
    
    # 3. Non-Local Means Denoising
    print("Running Non-Local Means Denoising...")
    # Estimate the Rician noise standard deviation (standard for MRI)
    sigma = estimate_sigma(n4_array, N=32)
    
    # Apply the edge-preserving NLM filter
    denoised_array = nlmeans(n4_array, 
                             sigma=sigma, 
                             mask=(n4_array > 0), 
                             patch_radius=1, 
                             block_radius=2, 
                             rician=True)
    
    # 4. Save the Restored Image
    denoised_image = sitk.GetImageFromArray(denoised_array)
    denoised_image.CopyInformation(n4_image) # Preserve original headers & spacing
    sitk.WriteImage(denoised_image, output_path)
    print(f"Saved restored image to: {output_path}\n")

# Process the modalities for Patient 100101A
modalities = ["T1pre", "T1post", "FLAIR", "T2Synth"]
for mod in modalities:
    input_file = f"data/UCSF_BrainMetastases_TRAIN/100101A/100101A_{mod}.nii.gz"
    output_file = f"data/UCSF_BrainMetastases_TRAIN/100101A/100101A_{mod}_Restored.nii.gz"
    restore_mri(input_file, output_file)