import json
import shutil
from pathlib import Path

INPUT_DIR = Path("data/UCSF_BrainMetastases_TRAIN")
# Changed Dataset ID from 100 to 102 so you don't overwrite your baseline data
OUTPUT_DIR = Path("nnunet_data/nnUNet_raw/Dataset102_Restored")

IMAGES_TR = OUTPUT_DIR / "imagesTr"
LABELS_TR = OUTPUT_DIR / "labelsTr"

def main():
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} total subject folders. Starting conversion...")

    converted_count = 0

    for pat_dir in patient_dirs:
        pat_id = pat_dir.name
        
        # Point to the newly generated Restored files
        t1pre = pat_dir / f"{pat_id}_T1pre_Restored.nii.gz"
        t1post = pat_dir / f"{pat_id}_T1post_Restored.nii.gz"
        flair = pat_dir / f"{pat_id}_FLAIR_Restored.nii.gz"
        
        # Check T2Synth Restored first, fallback to native T2 Restored
        t2 = pat_dir / f"{pat_id}_T2Synth_Restored.nii.gz"
        if not t2.exists():
            t2 = pat_dir / f"{pat_id}_T2_Restored.nii.gz"
            
        # Ground truth labels remain exactly the same (we do not denoise the masks!)
        label_file = pat_dir / f"{pat_id}_BraTS-seg.nii.gz"
        if not label_file.exists():
            label_file = pat_dir / f"{pat_id}_seg.nii.gz"

        # Verify all 4 restored modalities and label exist
        modalities = [t1pre, t1post, t2, flair]
        if not all(m.exists() for m in modalities) or not label_file.exists():
            print(f"Skipping {pat_id}: Incomplete restored modalities or missing segmentation.")
            continue

        # Target destinations formatted for nnU-Net
        shutil.copy(t1pre, IMAGES_TR / f"{pat_id}_0000.nii.gz")
        shutil.copy(t1post, IMAGES_TR / f"{pat_id}_0001.nii.gz")
        shutil.copy(t2, IMAGES_TR / f"{pat_id}_0002.nii.gz")
        shutil.copy(flair, IMAGES_TR / f"{pat_id}_0003.nii.gz")
        shutil.copy(label_file, LABELS_TR / f"{pat_id}.nii.gz")

        converted_count += 1

    dataset_json = {
        "channel_names": {
            "0": "T1pre",
            "1": "T1post",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "background": 0,
            "core": 1,
            "edema": 2,
            "enhancing": 3
        },
        "numTraining": converted_count,
        "file_ending": ".nii.gz"
    }

    with open(OUTPUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n[✓] Successfully converted {converted_count} subjects into nnU-Net format!")
    print(f"[✓] dataset.json written to {OUTPUT_DIR / 'dataset.json'}")

if __name__ == "__main__":
    main()