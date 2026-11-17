"""
evaluate.py
-----------
Loads the trained 3D U-Net checkpoint, runs full-volume inference on the
test split, computes test metrics, and exports comparison slice figures.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import numpy as np
import torch

from src.model import build_3d_unet
from src.transforms import get_transforms

# 1. Configuration
SPLIT_JSON = Path("data/dataset_split.json")
CHECKPOINT_PATH = Path("checkpoints/best_3d_unet.pth")
OUTPUT_DIR = Path("results/figures")
PATCH_SIZE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  # 2. Load the test split
  with open(SPLIT_JSON, "r") as f:
    splits = json.load(f)
  test_files = splits["test"]
  print(f"Loaded {len(test_files)} unseen test cases.")

  # 3. Load model and checkpoint
  model = build_3d_unet(in_channels=4, out_channels=4).to(DEVICE)
  model.load_state_dict(
      torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
  )
  model.eval()
  print(f"Loaded weights from {CHECKPOINT_PATH}")

  # 4. Prepare test DataLoader
  _, val_transforms = get_transforms(patch_size=PATCH_SIZE)
  test_ds = Dataset(data=test_files, transform=val_transforms)
  test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

  dice_metric = DiceMetric(
      include_background=False, reduction="mean_batch"
  )  # Per-class Dice
  post_pred = AsDiscrete(argmax=True, to_onehot=4)
  post_label = AsDiscrete(to_onehot=4)

  print("\n--- Running Evaluation on Test Set ---")
  with torch.no_grad():
    for idx, test_data in enumerate(test_loader):
      inputs = test_data["image"].to(DEVICE)
      labels = test_data["label"].to(DEVICE)
      subj_id = test_files[idx]["subject_id"]

      # Full 3D sliding window inference
      outputs = sliding_window_inference(
          inputs=inputs,
          roi_size=PATCH_SIZE,
          sw_batch_size=4,
          predictor=model,
          overlap=0.5,
      )

      # Metric computation
      outputs_discrete = [post_pred(i) for i in decollate_batch(outputs)]
      labels_discrete = [post_label(i) for i in decollate_batch(labels)]
      dice_metric(y_pred=outputs_discrete, y=labels_discrete)

      # 5. Extract 3D volume arrays for 2D slice visualization
      t1post_np = inputs[0, 1].cpu().numpy()  # Channel 1 is T1post
      gt_np = labels[0, 0].cpu().numpy()  # Ground truth label volume
      pred_np = torch.argmax(outputs[0], dim=0).cpu().numpy()  # Predicted mask

      # Find the 2D axial slice containing the largest tumor area
      tumor_slice_idx = np.argmax(np.sum(gt_np > 0, axis=(0, 1)))

      # Export visual comparison figure if tumor exists in the slice
      if np.sum(gt_np[:, :, tumor_slice_idx]) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # (a) Input T1-Post MRI
        axes[0].imshow(t1post_np[:, :, tumor_slice_idx], cmap="gray")
        axes[0].set_title(f"{subj_id} - T1post (Slice {tumor_slice_idx})")
        axes[0].axis("off")

        # (b) Ground Truth Annotation Overlay
        axes[1].imshow(t1post_np[:, :, tumor_slice_idx], cmap="gray")
        gt_overlay = np.ma.masked_where(
            gt_np[:, :, tumor_slice_idx] == 0, gt_np[:, :, tumor_slice_idx]
        )
        axes[1].imshow(
            gt_overlay, cmap="autumn", alpha=0.7, vmin=1, vmax=3
        )  #
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        # (c) Model Prediction Overlay
        axes[2].imshow(t1post_np[:, :, tumor_slice_idx], cmap="gray")
        pred_overlay = np.ma.masked_where(
            pred_np[:, :, tumor_slice_idx] == 0, pred_np[:, :, tumor_slice_idx]
        )
        axes[2].imshow(pred_overlay, cmap="autumn", alpha=0.7, vmin=1, vmax=3)
        axes[2].set_title("3D U-Net Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"{subj_id}_comparison.png"
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved visualization figure: {save_path}")

  # 6. Print Final Quantitative Test Metrics
  test_dice_per_class = dice_metric.aggregate().cpu().numpy()
  print("\n================ TEST SET RESULTS ================")
  print(
      f"Label 1 (Non-enhancing core) Dice : {test_dice_per_class[0]:.4f}"
  )  #[cite: 1]
  print(
      f"Label 2 (Peritumoral edema)   Dice : {test_dice_per_class[1]:.4f}"
  )  #[cite: 1]
  print(
      f"Label 3 (Enhancing tumor)     Dice : {test_dice_per_class[2]:.4f}"
  )  #[cite: 1]
  print(
      f"Overall Mean Dice Score            : {np.mean(test_dice_per_class):.4f}"
  )
  print("==================================================")


if __name__ == "__main__":
  main()