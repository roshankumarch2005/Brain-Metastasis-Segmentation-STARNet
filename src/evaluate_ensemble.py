"""
evaluate_ensemble.py
--------------------
Averages the predictions of 3D U-Net and SegResNet.
"""
import json
import torch
import numpy as np
from pathlib import Path
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from src.model import build_3d_unet, build_segresnet
from src.transforms import get_transforms

SPLIT_JSON = Path("data/dataset_split.json")
PATCH_SIZE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(SPLIT_JSON, "r") as f:
        test_files = json.load(f)["test"]

    # Load both models
    unet = build_3d_unet(in_channels=4, out_channels=4).to(DEVICE)
    unet.load_state_dict(torch.load("checkpoints/best_3d_unet.pth", map_location=DEVICE, weights_only=True))
    unet.eval()

    segresnet = build_segresnet(in_channels=4, out_channels=4).to(DEVICE)
    segresnet.load_state_dict(torch.load("checkpoints/best_segresnet.pth", map_location=DEVICE, weights_only=True))
    segresnet.eval()

    _, val_transforms = get_transforms(patch_size=PATCH_SIZE)
    test_loader = DataLoader(Dataset(data=test_files, transform=val_transforms), batch_size=1)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred, post_label = AsDiscrete(argmax=True, to_onehot=4), AsDiscrete(to_onehot=4)

    print("\n--- Running Ensemble Evaluation ---")
    with torch.no_grad():
        for test_data in test_loader:
            inputs, labels = test_data["image"].to(DEVICE), test_data["label"].to(DEVICE)

            # Get raw predictions from both models
            out_unet = sliding_window_inference(inputs, PATCH_SIZE, 4, unet, 0.5)
            out_seg = sliding_window_inference(inputs, PATCH_SIZE, 4, segresnet, 0.5)

            # Average the raw logits/probabilities
            ensemble_out = (out_unet + out_seg) / 2.0

            dice_metric(y_pred=[post_pred(i) for i in decollate_batch(ensemble_out)], 
                        y=[post_label(i) for i in decollate_batch(labels)])

    metrics = dice_metric.aggregate().cpu().numpy()
    print("\n========= ENSEMBLE TEST SET RESULTS =========")
    print(f"Core (1) Dice     : {metrics[0]:.4f}")
    print(f"Edema (2) Dice    : {metrics[1]:.4f}")
    print(f"Enhancing (3) Dice: {metrics[2]:.4f}")
    print(f"Overall Mean Dice : {np.mean(metrics):.4f}")
    print("=============================================")

if __name__ == "__main__":
    main()
