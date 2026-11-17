import os
import torch
import torch.nn as nn
import numpy as np
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
from nnunetv2.training.nnUNetTrainer.models.siam import SIAM3D, SIAM2D

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D
except ImportError:
    try:
        from .nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D
    except ImportError:
        from nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D


class CurriculumLossWrapper(nn.Module):
    """
    Two-Stage Curriculum Loss Wrapper:
    - Stage 1 (Epochs 0 to 499): Default DC + CE Loss (stable representation anchor)
    - Stage 2 (Epochs 500 to 1000): Area-Aware BAA Sensitive Loss beta=1.5 (fine boundary & small lesion refinement)
    """
    def __init__(self, loss_stage1, loss_stage2, trainer, switch_epoch=500):
        super().__init__()
        self.loss_stage1 = loss_stage1
        self.loss_stage2 = loss_stage2
        self.trainer = trainer
        self.switch_epoch = switch_epoch
        self.has_logged_switch = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self.loss_stage1, 'to'):
            self.loss_stage1.to(*args, **kwargs)
        if hasattr(self.loss_stage2, 'to'):
            self.loss_stage2.to(*args, **kwargs)
        return self

    def forward(self, *args):
        if self.trainer.current_epoch < self.switch_epoch:
            return self.loss_stage1(*args)
        else:
            if not self.has_logged_switch:
                if not hasattr(self.trainer, 'local_rank') or self.trainer.local_rank == 0:
                    self.trainer.print_to_log_file(
                        f"\n{'='*70}\n"
                        f"[CURRICULUM SWITCH] Transitioning to Stage 2 (Epoch {self.trainer.current_epoch})\n"
                        f"Loss Function: Volume-Adaptive BAA Loss (beta=1.5)\n"
                        f"{'='*70}\n"
                    )
                self.has_logged_switch = True
            return self.loss_stage2(*args)


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


class nnUNetTrainer_Method3_SIAM_CurriculumBAA(nnUNetTrainer_DepthwiseSeparable3D):
    """
    Synergistic Model: Method 3 (3D Depthwise-Separable Convolutions, 1.49M params)
                     + SIAM3D Spatial Attention
                     + Two-Stage Curriculum BAA (Epoch 0-499: DC+CE -> Epoch 500-1000: BAA beta=1.5).
    
    Trained on Dataset102_Restored for Multi-Class BraTS segmentation (NCR, ED, ET).
    Target: Deliver balanced multi-compartment recall on an ultra-compact ~11 MB footprint.
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
        # 1. Stage 1 Loss: Default Compound DC + CE Loss
        loss_default = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # 2. Stage 2 Loss: Volume-Adaptive BAA Sensitive Loss (beta=1.5)
        num_classes = self.label_manager.num_segmentation_heads
        if num_classes == 2:
            class_freqs = [0.992, 0.008]
        elif num_classes == 3:
            class_freqs = [0.950, 0.045, 0.005]
        elif num_classes == 4:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        else:
            class_freqs = [0.98] + [0.02 / (num_classes - 1)] * (num_classes - 1)

        loss_baa = SmoothedBAAClassBalanced_DC_and_CE_loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            beta_scale=1.5,
            area_threshold=5000,
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()

        wrapped_stage1 = DeepSupervisionWrapper(loss_default, weights)
        wrapped_stage2 = DeepSupervisionWrapper(loss_baa, weights)

        return CurriculumLossWrapper(wrapped_stage1, wrapped_stage2, self, switch_epoch=500)

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision: bool = True):
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        
        # Base 16, Cap 256 feature progression (slashing parameters by ~96% to ~1.49M)
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
