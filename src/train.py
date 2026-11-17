"""
train.py
--------
Training pipeline for 3D Brain Metastases Segmentation using MONAI & PyTorch.
"""

import json
import os
from pathlib import Path
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import torch

from src.model import build_3d_unet
from src.transforms import get_transforms

# 1. Configuration
SPLIT_JSON = Path("data/dataset_split.json")
PATCH_SIZE = (96, 96, 96)
BATCH_SIZE = 2  # 2 volumes * 4 patches = 8 3D patches per training step
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
MAX_EPOCHS = 60
VAL_INTERVAL = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
  print(f"Using device: {DEVICE}")
  if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

  # 2. Load dataset splits
  with open(SPLIT_JSON, "r") as f:
    splits = json.load(f)

  train_files = splits["train"]
  val_files = splits["val"]

  # 3. Transforms and Datasets
  train_transforms, val_transforms = get_transforms(patch_size=PATCH_SIZE)

  train_ds = Dataset(data=train_files, transform=train_transforms)
  # list_data_collate collates the multi-patch outputs into standard batch tensors
  train_loader = DataLoader(
      train_ds,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=NUM_WORKERS,
      collate_fn=list_data_collate,
      pin_memory=torch.cuda.is_available(),
  )

  val_ds = Dataset(data=val_files, transform=val_transforms)
  val_loader = DataLoader(
      val_ds,
      batch_size=1,
      num_workers=NUM_WORKERS,
      pin_memory=torch.cuda.is_available(),
  )

  # 4. Model, Loss, Optimizer, Metric
  model = build_3d_unet(in_channels=4, out_channels=4).to(DEVICE)
  loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
  )

  dice_metric = DiceMetric(include_background=False, reduction="mean")
  post_pred = AsDiscrete(argmax=True, to_onehot=4)
  post_label = AsDiscrete(to_onehot=4)

  best_val_dice = -1.0
  os.makedirs("checkpoints", exist_ok=True)

  print("\n--- Starting Training ---")
  for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    step = 0

    for batch_data in train_loader:
      step += 1
      inputs = batch_data["image"].to(DEVICE)
      labels = batch_data["label"].to(DEVICE)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

      if step % 20 == 0 or step == len(train_loader):
        print(
            f"Epoch [{epoch}/{MAX_EPOCHS}] - Step [{step}/{len(train_loader)}]"
            f" - Patch Loss: {loss.item():.4f}"
        )

    epoch_loss /= step
    print(f"--> Epoch [{epoch}/{MAX_EPOCHS}] Completed | Avg Loss: {epoch_loss:.4f}")

    # 5. Validation Loop
    if epoch % VAL_INTERVAL == 0:
      model.eval()
      with torch.no_grad():
        for val_data in val_loader:
          val_inputs = val_data["image"].to(DEVICE)
          val_labels = val_data["label"].to(DEVICE)

          # Sliding window inference evaluates the full 3D volume seamlessly
          val_outputs = sliding_window_inference(
              inputs=val_inputs,
              roi_size=PATCH_SIZE,
              sw_batch_size=4,
              predictor=model,
              overlap=0.5,
          )

          val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
          val_labels = [post_label(i) for i in decollate_batch(val_labels)]
          dice_metric(y_pred=val_outputs, y=val_labels)

        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        print(f"\n[Validation] Mean Dice Score: {val_dice:.4f}")

        if val_dice > best_val_dice:
          best_val_dice = val_dice
          torch.save(model.state_dict(), "checkpoints/best_3d_unet.pth")
          print(
              f"[*] New Best Model Saved to checkpoints/best_3d_unet.pth (Dice:"
              f" {best_val_dice:.4f})\n"
          )


if __name__ == "__main__":
  main()