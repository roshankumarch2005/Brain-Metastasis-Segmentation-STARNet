import os
import json
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

INPUT_DIR = Path("data/UCSF_BrainMetastases_TRAIN")
OUTPUT_DIR = Path("nnunet_data/nnUNet_raw/Dataset106_UCSF_Saluja_Restored")
SPLIT_JSON = Path("data/dataset_split.json")

IMAGES_TR = OUTPUT_DIR / "imagesTr"
LABELS_TR = OUTPUT_DIR / "labelsTr"
IMAGES_TS = OUTPUT_DIR / "imagesTs"
LABELS_TS = OUTPUT_DIR / "labelsTs"

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)
IMAGES_TS.mkdir(parents=True, exist_ok=True)
LABELS_TS.mkdir(parents=True, exist_ok=True)

with open(SPLIT_JSON, "r") as f:
    test_entries = json.load(f)["test"]
test_ids = set([entry["subject_id"] if isinstance(entry, dict) else Path(entry).parent.name for entry in test_entries])
print(f"Identified {len(test_ids)} held-out test subjects.")

def process_patient(pat_dir):
    pat_id = pat_dir.name
    tr_0 = IMAGES_TR / f"{pat_id}_0000.nii.gz"
    tr_1 = IMAGES_TR / f"{pat_id}_0001.nii.gz"
    tr_2 = IMAGES_TR / f"{pat_id}_0002.nii.gz"
    tr_3 = IMAGES_TR / f"{pat_id}_0003.nii.gz"
    tr_lbl = LABELS_TR / f"{pat_id}.nii.gz"

    if tr_0.exists() and tr_1.exists() and tr_2.exists() and tr_3.exists() and tr_lbl.exists():
        if pat_id in test_ids and not (IMAGES_TS / f"{pat_id}_0000.nii.gz").exists():
            shutil.copy(tr_0, IMAGES_TS / f"{pat_id}_0000.nii.gz")
            shutil.copy(tr_1, IMAGES_TS / f"{pat_id}_0001.nii.gz")
            shutil.copy(tr_2, IMAGES_TS / f"{pat_id}_0002.nii.gz")
            shutil.copy(tr_3, IMAGES_TS / f"{pat_id}_0003.nii.gz")
            shutil.copy(tr_lbl, LABELS_TS / f"{pat_id}.nii.gz")
        return (pat_id, True, "already exists")

    t1post_r = pat_dir / f"{pat_id}_T1post_Restored.nii.gz"
    t1pre_r = pat_dir / f"{pat_id}_T1pre_Restored.nii.gz"
    flair_r = pat_dir / f"{pat_id}_FLAIR_Restored.nii.gz"
    label_file = pat_dir / f"{pat_id}_seg.nii.gz"

    if not (t1post_r.exists() and t1pre_r.exists() and flair_r.exists() and label_file.exists()):
        return (pat_id, False, "missing files")

    try:
        post_img = nib.load(str(t1post_r))
        pre_img = nib.load(str(t1pre_r))
        sub_data = post_img.get_fdata().astype(np.float32) - pre_img.get_fdata().astype(np.float32)
        sub_img = nib.Nifti1Image(sub_data, post_img.affine, post_img.header)

        shutil.copy(t1post_r, tr_0)
        shutil.copy(t1pre_r, tr_1)
        nib.save(sub_img, tr_2)
        shutil.copy(flair_r, tr_3)
        shutil.copy(label_file, tr_lbl)

        if pat_id in test_ids:
            shutil.copy(tr_0, IMAGES_TS / f"{pat_id}_0000.nii.gz")
            shutil.copy(tr_1, IMAGES_TS / f"{pat_id}_0001.nii.gz")
            shutil.copy(tr_2, IMAGES_TS / f"{pat_id}_0002.nii.gz")
            shutil.copy(tr_3, IMAGES_TS / f"{pat_id}_0003.nii.gz")
            shutil.copy(label_file, LABELS_TS / f"{pat_id}.nii.gz")

        return (pat_id, True, "success")
    except Exception as e:
        print(f"Error on {pat_id}: {e}")
        # If restored file failed, fallback to raw T1post/T1pre/subtraction for this 1 patient
        try:
            print(f"Attempting fallback to raw for {pat_id}...")
            raw_post = pat_dir / f"{pat_id}_T1post.nii.gz"
            raw_pre = pat_dir / f"{pat_id}_T1pre.nii.gz"
            raw_sub = pat_dir / f"{pat_id}_subtraction.nii.gz"
            raw_flair = pat_dir / f"{pat_id}_FLAIR.nii.gz"
            shutil.copy(raw_post, tr_0)
            shutil.copy(raw_pre, tr_1)
            shutil.copy(raw_sub, tr_2)
            shutil.copy(raw_flair, tr_3)
            shutil.copy(label_file, tr_lbl)
            return (pat_id, True, "fallback raw")
        except Exception as e2:
            return (pat_id, False, f"failed raw fallback: {e2}")

def main():
    patient_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    print(f"Processing {len(patient_dirs)} patients using 16 parallel workers...")

    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_patient, patient_dirs))

    converted_tr = sum(1 for pid, ok, msg in results if ok)
    failed = [pid for pid, ok, msg in results if not ok]
    if failed:
        print(f"Failed patients: {failed}")

    dataset_json = {
        "channel_names": {
            "0": "T1post_Restored",
            "1": "T1pre_Restored",
            "2": "subtraction_Restored",
            "3": "FLAIR_Restored"
        },
        "labels": {
            "background": 0,
            "enhancing": 1
        },
        "numTraining": converted_tr,
        "file_ending": ".nii.gz"
    }

    with open(OUTPUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n[✓] Successfully built Dataset106_UCSF_Saluja_Restored!")
    print(f"    - Training cases (imagesTr): {converted_tr}")
    print(f"    - Held-out test cases (imagesTs): {len(test_ids)}")
    print(f"    - dataset.json written to {OUTPUT_DIR / 'dataset.json'}")

if __name__ == "__main__":
    main()
