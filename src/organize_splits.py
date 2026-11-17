import os
import json
import shutil
from glob import glob

base_dir = "data/UCSF_2D_Slices"
json_path = "data/dataset_split.json"

# Load the split definitions
with open(json_path, 'r') as f:
    splits = json.load(f)

# Map every subject ID to its split (e.g., "100108A" -> "train")
subj_to_split = {}
for split_name, patients in splits.items():
    for p in patients:
        subj_to_split[p["subject_id"]] = split_name

# Create the target folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

# Move the extracted .npy files into their correct split folders
for folder in ["images", "masks"]:
    src_folder = os.path.join(base_dir, folder)
    files = glob(os.path.join(src_folder, "*.npy"))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        subj_id = filename.split("_")[0]  # Extracts "100108A" from "100108A_slice_045.npy"
        
        if subj_id in subj_to_split:
            split_name = subj_to_split[subj_id]
            dest_path = os.path.join(base_dir, split_name, folder, filename)
            shutil.move(file_path, dest_path)

# Remove the now-empty original folders
try:
    os.rmdir(os.path.join(base_dir, "images"))
    os.rmdir(os.path.join(base_dir, "masks"))
except OSError:
    pass

print("Files successfully organized into train, val, and test folders!")
