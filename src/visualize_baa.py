import os
import nibabel as nib
import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes

# Start a virtual display so the headless Linux server doesn't crash
pv.start_xvfb()

import sys

def generate_full_overlay(patient_id, pred_dir, output_suffix):
    print(f"Loading volumes for patient {patient_id} from {pred_dir}...")
    
    # 1. Load Background Anatomy (T1/FLAIR)
    mri_vol = nib.load(f"/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTs/{patient_id}_0000.nii.gz").get_fdata()
    
    # 2. Load Ground Truth (labelsTs)
    gt_vol = nib.load(f"/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/nnUNet_raw_data/Task100_BraTS/labelsTs/{patient_id}.nii.gz").get_fdata()
    
    # 3. Load prediction
    pred_vol = nib.load(f"{pred_dir}/{patient_id}.nii.gz").get_fdata()

    def get_pyvista_mesh(volume):
        if not np.any(volume):
            return pv.PolyData()
            
        verts, faces, _, _ = marching_cubes(volume, level=0.5)
        padded_faces = np.insert(faces, 0, 3, axis=1).flatten()
        return pv.PolyData(verts, padded_faces)

    print(f"Extracting meshes for ENHANCING TUMOR (Class 3) specifically for {patient_id}...")
    
    brain_threshold = 0.05 * mri_vol.max()
    brain_mesh = get_pyvista_mesh(mri_vol > brain_threshold)
    
    gt_mesh = get_pyvista_mesh(gt_vol == 3)
    pred_mesh = get_pyvista_mesh(pred_vol == 3)

    plotter = pv.Plotter(off_screen=True)
    
    if brain_mesh.n_points > 0:
        plotter.add_mesh(brain_mesh, color="lightgrey", opacity=0.1, label="Brain Anatomy")
    if gt_mesh.n_points > 0:
        plotter.add_mesh(gt_mesh, color="green", opacity=0.4, label="True Enhancing Tumor")
    if pred_mesh.n_points > 0:
        plotter.add_mesh(pred_mesh, color="red", opacity=1.0, label="Prediction")
        
    plotter.add_legend()
    plotter.set_background("white")
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.5)

    output_path = f"/mnt/DATA/EE23B009/EE23B009_BTP/results/figures_nnunet_overlay/{patient_id}_Enhancing_Tumor_{output_suffix}.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.export_html(output_path) 
    print(f"Success! Open {output_path} in your browser.")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        patient_id = sys.argv[1]
        pred_dir = sys.argv[2]
        output_suffix = sys.argv[3]
        generate_full_overlay(patient_id, pred_dir, output_suffix)
    else:
        print("Usage: python3 visualize_baa.py <patient_id> <pred_dir> <output_suffix>")
