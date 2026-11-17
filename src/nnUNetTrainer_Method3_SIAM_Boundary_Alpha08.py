import os
import torch
import torch.nn as nn
import numpy as np
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.losses.boundary_baa_loss import BoundaryAware_BAA_Loss
from nnunetv2.training.nnUNetTrainer.models.siam import SIAM3D, SIAM2D
try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D
except ImportError:
    try:
        from .nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D
    except ImportError:
        from nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D


class SIAMEncoderWrapper(nn.Module):
    """
    Wraps standard or depthwise PlainConvUNet encoder to apply SIAM3D attention
    at all skip connections and at the deepest bottleneck.
    """
    def __init__(self, original_encoder, is_3d: bool = True):
        super().__init__()
        self.original_encoder = original_encoder
        self.siam = SIAM3D() if is_3d else SIAM2D()

    @property
    def output_channels(self):
        return self.original_encoder.output_channels

    @property
    def strides(self):
        return self.original_encoder.strides

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_encoder, name)

    def forward(self, x):
        skips = self.original_encoder(x)
        if isinstance(skips, (list, tuple)):
            new_skips = [self.siam(skip) for skip in skips]
            if isinstance(skips, tuple):
                return tuple(new_skips)
            return new_skips
        return self.siam(skips)


class nnUNetTrainer_Method3_SIAM_Boundary_Alpha08(nnUNetTrainer_DepthwiseSeparable3D):
    """
    Synergistic Model: Method 3 (3D Depthwise-Separable Convolutions, 3.8M params)
                     + SIAM3D Spatial Attention
                     + Convex 80/20 Boundary-Aware Loss (Euclidean Distance Transform).
    
    Trained on Dataset105_UCSF_Saluja_Exact for Binary GTV segmentation.
    Target: Deliver single-model 5.5mm boundary precision on an ultra-compact ~11 MB footprint.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000

    def initialize(self):
        super().initialize()
        if hasattr(self, 'loss') and hasattr(self.loss, 'to'):
            self.loss.to(self.device)

    def _build_loss(self):
        return self.build_loss()

    def build_loss(self):
        num_classes = self.label_manager.num_segmentation_heads
        if num_classes == 2:
            class_freqs = [0.992, 0.008]
        elif num_classes == 3:
            class_freqs = [0.950, 0.045, 0.005]
        elif num_classes == 4:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        else:
            class_freqs = [0.98] + [0.02 / (num_classes - 1)] * (num_classes - 1)
        
        # Convex 80/20 Boundary Loss (The engine of Model B3 - 5.58 mm HD95):
        loss = BoundaryAware_BAA_Loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            boundary_scale=1.5,
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_dice=0.8,  # Volume preservation priority (Dice 80%)
            weight_ce=0.2,    # Boundary penalty priority (CE 20%)
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        
        return DeepSupervisionWrapper(loss, weights)

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision: bool = True):
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        
        # Base 16, Cap 256 feature progression (slashing parameters by ~88% to ~3.8M)
        base_features = 16
        max_features = 256
        n_stages = len(configuration_manager.conv_kernel_sizes)
        features_per_stage = [min(base_features * (2 ** i), max_features) for i in range(n_stages)]
        
        kernel_sizes = configuration_manager.conv_kernel_sizes
        strides = configuration_manager.pool_op_kernel_sizes
        n_conv_per_stage = configuration_manager.n_conv_per_stage_encoder
        n_conv_per_stage_decoder = configuration_manager.n_conv_per_stage_decoder
        
        network = PlainConvUNet(
            input_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=DepthwiseSeparableConv3d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
            deep_supervision=enable_deep_supervision
        )
        
        # Inject SIAM attention into encoder
        is_3d = len(configuration_manager.patch_size) == 3
        if hasattr(network, 'encoder'):
            network.encoder = SIAMEncoderWrapper(network.encoder, is_3d=is_3d)
            
        return network
