"""
prepare_dataset100_fast.py
-------------------------
Fast dataset assembler for Dataset100_BraTS (Raw 4-channel Multi-Class BraTS).
Uses symlinks for instant, zero-copy assembly in under 3 seconds!

Input Modalities (4 channels):
  0: T1pre
  1: T1post
  2: T2Synth (or T2)
  3: FLAIR
Target:
  labelsTr: BraTS-seg.nii.gz (1: Necrotic Core, 2: Edema, 3: Enhancing Tumor)
"""

import os
import json
from pathlib import Path

BASE_DIR = Path("/Data3/EE23B009/EE23B009_BTP") if Path("/Data3/EE23B009/EE23B009_BTP").exists() else Path("/mnt/DATA/EE23B009/EE23B009_BTP")
INPUT_DIR = BASE_DIR / "data/UCSF_BrainMetastases_TRAIN"
OUTPUT_DIR = BASE_DIR / "nnunet_data/nnUNet_raw/Dataset100_BraTS"
SPLIT_JSON = BASE_DIR / "data/dataset_split.json"

IMAGES_TR = OUTPUT_DIR / "imagesTr"
LABELS_TR = OUTPUT_DIR / "labelsTr"
IMAGES_TS = OUTPUT_DIR / "imagesTs"
LABELS_TS = OUTPUT_DIR / "labelsTs"

def main():
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)
    IMAGES_TS.mkdir(parents=True, exist_ok=True)
    LABELS_TS.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_JSON, "r") as f:
        test_entries = json.load(f)["test"]
    test_ids = set([entry["subject_id"] if isinstance(entry, dict) else Path(entry).parent.name for entry in test_entries])
    print(f"Identified {len(test_ids)} held-out test subjects.")

    patient_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} total subject folders.")

    count_tr = 0
    count_ts = 0

    for pat_dir in patient_dirs:
        pat_id = pat_dir.name

        t1pre = pat_dir / f"{pat_id}_T1pre.nii.gz"
        t1post = pat_dir / f"{pat_id}_T1post.nii.gz"
        t2 = pat_dir / f"{pat_id}_T2Synth.nii.gz"
        if not t2.exists():
            t2 = pat_dir / f"{pat_id}_T2.nii.gz"
        flair = pat_dir / f"{pat_id}_FLAIR.nii.gz"
        
        label_file = pat_dir / f"{pat_id}_BraTS-seg.nii.gz"
        if not label_file.exists():
            label_file = pat_dir / f"{pat_id}_seg.nii.gz"

        modalities = [t1pre, t1post, t2, flair]
        if not all(m.exists() for m in modalities) or not label_file.exists():
            print(f"Skipping {pat_id}: Missing raw modalities or label.")
            continue

        # Fast relative or absolute symlinks
        for idx, mod_path in enumerate(modalities):
            dest = IMAGES_TR / f"{pat_id}_{idx:04d}.nii.gz"
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(mod_path.resolve())

        lbl_dest = LABELS_TR / f"{pat_id}.nii.gz"
        if lbl_dest.exists() or lbl_dest.is_symlink():
            lbl_dest.unlink()
        lbl_dest.symlink_to(label_file.resolve())
        count_tr += 1

        if pat_id in test_ids:
            for idx, mod_path in enumerate(modalities):
                dest_ts = IMAGES_TS / f"{pat_id}_{idx:04d}.nii.gz"
                if dest_ts.exists() or dest_ts.is_symlink():
                    dest_ts.unlink()
                dest_ts.symlink_to(mod_path.resolve())
            lbl_ts = LABELS_TS / f"{pat_id}.nii.gz"
            if lbl_ts.exists() or lbl_ts.is_symlink():
                lbl_ts.unlink()
            lbl_ts.symlink_to(label_file.resolve())
            count_ts += 1

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
        "numTraining": count_tr,
        "file_ending": ".nii.gz"
    }

    with open(OUTPUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Successfully assembled Dataset100_BraTS!")
    print(f"  • Training cases (imagesTr): {count_tr}")
    print(f"  • Test cases (imagesTs):     {count_ts}")

if __name__ == "__main__":
    main()
