import os
import nibabel as nib
import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes

# Start a virtual display so the headless Linux server doesn't crash
pv.start_xvfb()

def generate_full_overlay():
    print("Loading volumes...")
    
    mri_vol = nib.load("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTr/100362A_0000.nii.gz").get_fdata()
    gt_vol = nib.load("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/labelsTr/100362A.nii.gz").get_fdata()
    pred_vol = nib.load("/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/predictions_test_cleaned/100362A.nii.gz").get_fdata()

    def get_pyvista_mesh(volume, threshold_val=0):
        if not np.any(volume > threshold_val):
            return pv.PolyData()
            
        verts, faces, _, _ = marching_cubes(volume > threshold_val, level=0.5)
        padded_faces = np.insert(faces, 0, 3, axis=1).flatten()
        return pv.PolyData(verts, padded_faces)

    print("Extracting meshes (this takes a moment for the full brain)...")
    
    brain_threshold = 0.05 * mri_vol.max()
    
    brain_mesh = get_pyvista_mesh(mri_vol, threshold_val=brain_threshold)
    gt_mesh = get_pyvista_mesh(gt_vol)
    pred_mesh = get_pyvista_mesh(pred_vol)

    plotter = pv.Plotter(off_screen=True)
    
    if brain_mesh.n_points > 0:
        plotter.add_mesh(brain_mesh, color="lightgrey", opacity=0.15, label="Brain Anatomy")
    if gt_mesh.n_points > 0:
        plotter.add_mesh(gt_mesh, color="green", opacity=0.3, label="Ground Truth")
    if pred_mesh.n_points > 0:
        plotter.add_mesh(pred_mesh, color="red", opacity=1.0, label="Prediction")
        
    plotter.add_legend()
    plotter.set_background("white")

    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)

    output_path = "/mnt/DATA/EE23B009/EE23B009_BTP/results/figures_nnunet_overlay/100362A_Full_Brain_Overlay.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Updated to the modern PyVista API
    plotter.export_html(output_path) 
    print(f"Success! Open {output_path} in your browser.")

if __name__ == "__main__":
    generate_full_overlay()