"""
model.py
--------
Defines the segmentation network architectures for multi-modal brain MRI.
"""
from monai.networks.nets import UNet, SegResNet
import torch
import torch.nn as nn

def build_3d_unet(
    in_channels: int = 4,
    out_channels: int = 4,
) -> nn.Module:
    """Baseline 3D U-Net."""
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )
    return model

def build_segresnet(
    in_channels: int = 4,
    out_channels: int = 4,
) -> nn.Module:
    """
    State-of-the-Art (SOTA) SegResNet optimized for BraTS brain tumor segmentation.
    This architecture utilizes residual skip connections for superior gradient flow.
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=16,
        dropout_prob=0.2,
    )
    return model

if __name__ == "__main__":
    net = build_segresnet()
    dummy_x = torch.randn(1, 4, 96, 96, 96)
    out = net(dummy_x)
    print("SOTA SegResNet Output Shape:", out.shape)