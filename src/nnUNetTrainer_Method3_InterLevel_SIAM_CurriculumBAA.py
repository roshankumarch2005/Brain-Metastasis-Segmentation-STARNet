import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
from nnunetv2.training.nnUNetTrainer.models.siam import SIAM3D, SIAM2D

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d
except ImportError:
    try:
        from .nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d
    except ImportError:
        from nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_InterLevelAll import InterLevelUNetWrapper, nnUNetTrainer_InterLevelAll
except ImportError:
    try:
        from .nnUNetTrainer_InterLevelAll import InterLevelUNetWrapper, nnUNetTrainer_InterLevelAll
    except ImportError:
        from nnUNetTrainer_InterLevelAll import InterLevelUNetWrapper, nnUNetTrainer_InterLevelAll

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_InterLevel_SIAM import SIAMEncoderWrapper
except ImportError:
    try:
        from .nnUNetTrainer_InterLevel_SIAM import SIAMEncoderWrapper
    except ImportError:
        from nnUNetTrainer_InterLevel_SIAM import SIAMEncoderWrapper

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import CurriculumLossWrapper
except ImportError:
    try:
        from .nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import CurriculumLossWrapper
    except ImportError:
        from nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import CurriculumLossWrapper


class nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA(nnUNetTrainer_InterLevelAll):
    """
    Configuration 15: The Ultimate Unified Edge Architecture.
    
    Unites ALL Core Innovations:
    1. Method 3: 3D Depthwise-Separable Convolutions (1.49M params, -95% size)
    2. Spatial Intersection Attention Module (SIAM3D) on skip connections and bottleneck
    3. Method 1: Inter-Level Multi-Stage Hinge Regularization anchored to d0 (Tr = 0.20, lambda = 1.0)
    4. Two-Stage Curriculum Learning:
       - Stage 1 (Epochs 0 to 499): Compound DC + CE Loss
       - Stage 2 (Epochs 500 to 1000): Volume-Adaptive BAA Sensitive Loss (beta = 1.5)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        # Original Method 1 Hyperparameters
        self.Tr = 0.20
        self.inter_loss_weight = 1.0
        self.stage_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

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
        
        # Base 16, Cap 256 feature progression
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
        
        # 1. Apply SIAM3D attention along skip connections and bottleneck
        is_3d = len(configuration_manager.patch_size) == 3
        if hasattr(network, 'encoder'):
            network.encoder = SIAMEncoderWrapper(network.encoder, is_3d=is_3d)
            
        # 2. Attach Method 1 Bottleneck Seg Head and 6-stage wrapper
        wrapped_network = InterLevelUNetWrapper(network, num_classes=num_classes)
        return wrapped_network

