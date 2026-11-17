"""
prepare_nnunet_test.py
----------------------
Extracts the 30 test subjects from dataset_split.json and formats 
them into the nnU-Net imagesTs (Test) directory for inference.
"""
import json
import shutil
from pathlib import Path

SPLIT_JSON = Path("data/dataset_split.json")
OUTPUT_TS = Path("nnunet_data/nnUNet_raw/Dataset100_BraTS/imagesTs")
OUTPUT_GT = Path("nnunet_data/nnUNet_raw/Dataset100_BraTS/labelsTs")

def main():
    OUTPUT_TS.mkdir(parents=True, exist_ok=True)
    OUTPUT_GT.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_JSON, "r") as f:
        test_files = json.load(f)["test"]

    print(f"Found {len(test_files)} test subjects. Copying to imagesTs...")

    for case in test_files:
        # Assuming your JSON holds dictionaries like {"image": "...", "label": "..."}
        # or a list of paths. We extract the patient ID from the label path.
        label_path = Path(case["label"] if isinstance(case, dict) else case)
        pat_id = label_path.parent.name
        
        pat_dir = label_path.parent
        
        t1pre = pat_dir / f"{pat_id}_T1pre.nii.gz"
        t1post = pat_dir / f"{pat_id}_T1post.nii.gz"
        flair = pat_dir / f"{pat_id}_FLAIR.nii.gz"
        
        t2 = pat_dir / f"{pat_id}_T2Synth.nii.gz"
        if not t2.exists():
            t2 = pat_dir / f"{pat_id}_T2.nii.gz"

        shutil.copy(t1pre, OUTPUT_TS / f"{pat_id}_0000.nii.gz")
        shutil.copy(t1post, OUTPUT_TS / f"{pat_id}_0001.nii.gz")
        shutil.copy(t2, OUTPUT_TS / f"{pat_id}_0002.nii.gz")
        shutil.copy(flair, OUTPUT_TS / f"{pat_id}_0003.nii.gz")
        shutil.copy(label_path, OUTPUT_GT / f"{pat_id}.nii.gz")

    print("[✓] Test set perfectly formatted for nnU-Net!")

if __name__ == "__main__":
    main()