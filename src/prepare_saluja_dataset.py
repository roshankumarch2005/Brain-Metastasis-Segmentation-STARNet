import os
import json
import shutil
from pathlib import Path

INPUT_DIR = Path("data/UCSF_BrainMetastases_TRAIN")
OUTPUT_DIR = Path("nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact")
SPLIT_JSON = Path("data/dataset_split.json")

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
    print(f"Found {len(patient_dirs)} total subject folders. Processing conversion...")

    converted_tr = 0
    converted_ts = 0

    for pat_dir in patient_dirs:
        pat_id = pat_dir.name

        # Exact Saluja 4-modality inputs: T1post, T1pre, subtraction, FLAIR
        t1post = pat_dir / f"{pat_id}_T1post.nii.gz"
        t1pre = pat_dir / f"{pat_id}_T1pre.nii.gz"
        sub = pat_dir / f"{pat_id}_subtraction.nii.gz"
        flair = pat_dir / f"{pat_id}_FLAIR.nii.gz"
        
        # Saluja exact ground truth: seg.nii.gz (binary enhancing GTV)
        label_file = pat_dir / f"{pat_id}_seg.nii.gz"

        modalities = [t1post, t1pre, sub, flair]
        if not all(m.exists() for m in modalities) or not label_file.exists():
            print(f"Skipping {pat_id}: missing raw modalities or seg mask.")
            continue

        # Always copy to imagesTr (full 418 cohort for nnU-Net 5-fold CV)
        shutil.copy(t1post, IMAGES_TR / f"{pat_id}_0000.nii.gz")
        shutil.copy(t1pre, IMAGES_TR / f"{pat_id}_0001.nii.gz")
        shutil.copy(sub, IMAGES_TR / f"{pat_id}_0002.nii.gz")
        shutil.copy(flair, IMAGES_TR / f"{pat_id}_0003.nii.gz")
        shutil.copy(label_file, LABELS_TR / f"{pat_id}.nii.gz")
        converted_tr += 1

        # If it's part of the official 30 test cases, also populate imagesTs
        if pat_id in test_ids:
            shutil.copy(t1post, IMAGES_TS / f"{pat_id}_0000.nii.gz")
            shutil.copy(t1pre, IMAGES_TS / f"{pat_id}_0001.nii.gz")
            shutil.copy(sub, IMAGES_TS / f"{pat_id}_0002.nii.gz")
            shutil.copy(flair, IMAGES_TS / f"{pat_id}_0003.nii.gz")
            shutil.copy(label_file, LABELS_TS / f"{pat_id}.nii.gz")
            converted_ts += 1

    dataset_json = {
        "channel_names": {
            "0": "T1post",
            "1": "T1pre",
            "2": "subtraction",
            "3": "FLAIR"
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

    print(f"\n[✓] Successfully converted {converted_tr} training subjects into imagesTr!")
    print(f"[✓] Successfully converted {converted_ts} held-out test subjects into imagesTs!")
    print(f"[✓] Written dataset.json to {OUTPUT_DIR / 'dataset.json'}")

if __name__ == "__main__":
    main()
