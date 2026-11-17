import os
import sys
import nibabel as nib
import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes

pv.start_xvfb()

def generate_all_classes_overlay(patient_id):
    print(f"Loading volumes for patient {patient_id}...")
    
    # Background Anatomy
    mri_vol = nib.load(f"/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset102_Restored/imagesTr/{patient_id}_0000.nii.gz").get_fdata()
    # Prediction
    pred_vol = nib.load(f"/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/predictions_test_siam_baa/{patient_id}.nii.gz").get_fdata()

    def get_pyvista_mesh(volume):
        if not np.any(volume):
            return pv.PolyData()
        verts, faces, _, _ = marching_cubes(volume, level=0.5)
        padded_faces = np.insert(faces, 0, 3, axis=1).flatten()
        return pv.PolyData(verts, padded_faces)

    brain_threshold = 0.05 * mri_vol.max()
    brain_mesh = get_pyvista_mesh(mri_vol > brain_threshold)
    
    # Extract classes
    pred_ncr = get_pyvista_mesh(pred_vol == 1) # Necrotic Core
    pred_ed = get_pyvista_mesh(pred_vol == 2)  # Edema
    pred_et = get_pyvista_mesh(pred_vol == 3)  # Enhancing Tumor

    plotter = pv.Plotter(off_screen=True)
    
    if brain_mesh.n_points > 0:
        plotter.add_mesh(brain_mesh, color="lightgrey", opacity=0.05, label="Brain Anatomy")
        
    if pred_ed.n_points > 0:
        plotter.add_mesh(pred_ed, color="yellow", opacity=0.3, label="Edema (Class 2)")
    if pred_ncr.n_points > 0:
        plotter.add_mesh(pred_ncr, color="blue", opacity=0.5, label="Necrotic Core (Class 1)")
    if pred_et.n_points > 0:
        plotter.add_mesh(pred_et, color="red", opacity=0.3, label="Enhancing Tumor (Class 3)")
        
    plotter.add_legend()
    plotter.set_background("white")

    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.5)

    output_path = f"/mnt/DATA/EE23B009/EE23B009_BTP/results/figures_nnunet_overlay/{patient_id}_All_Classes_Pred.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plotter.export_html(output_path) 
    print(f"Success! Open {output_path} in your browser.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
        generate_all_classes_overlay(patient_id)
    else:
        print("Please provide a patient ID.")
