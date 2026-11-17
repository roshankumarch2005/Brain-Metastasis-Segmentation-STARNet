import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("./data/UCSF_BrainMetastases_TRAIN")
EXCEL_PATH = Path("./TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx")
OUTPUT_JSON = Path("./data/dataset_split.json")


def create_dataset_splits():
    # Read the metadata sheet
    df = pd.read_excel(EXCEL_PATH)

    # Filter only cases with valid BraTS IDs (the 324 pre-op cases)
    valid_subjects = df[df["BraTS_ID"].notna()]["SubjectID"].tolist()

    data_samples = []

    for subj in valid_subjects:
        subj_dir = DATA_DIR / subj
        t1pre = subj_dir / f"{subj}_T1pre.nii.gz"
        t1post = subj_dir / f"{subj}_T1post.nii.gz"
        flair = subj_dir / f"{subj}_FLAIR.nii.gz"
        t2synth = subj_dir / f"{subj}_T2Synth.nii.gz"
        seg = subj_dir / f"{subj}_BraTS-seg.nii.gz"

        # Verify all 4 modalities and target mask exist on disk
        if all(
            p.exists() for p in [t1pre, t1post, flair, t2synth, seg]
        ):  #[cite: 1]
            data_samples.append(
                {
                    "subject_id": subj,
                    "image": [
                        str(t1pre),
                        str(t1post),
                        str(flair),
                        str(t2synth),
                    ],  #[cite: 1]
                    "label": str(seg),
                }
            )

    print(f"Verified {len(data_samples)} complete cases with BraTS labels.")

    # Split into 80% Train, 10% Val, 10% Test
    train_data, val_test = train_test_split(
        data_samples, test_size=0.20, random_state=42
    )
    val_data, test_data = train_test_split(
        val_test, test_size=0.50, random_state=42
    )

    split_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(split_dict, f, indent=4)

    print(f"Dataset split saved to {OUTPUT_JSON}")
    print(
        f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}"
    )


if __name__ == "__main__":
    create_dataset_splits()