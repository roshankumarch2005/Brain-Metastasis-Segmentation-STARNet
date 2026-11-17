import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

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


class nnUNetTrainer_Method3_InterLevel_SIAM(nnUNetTrainer_InterLevelAll):
    """
    Configuration 13: Depthwise Separable 3D + Method 1 Inter-Level + SIAM Attention.
    
    Combines:
    - Method 3: 3D Depthwise Separable Convolutions (1.49M params, base 16, cap 256)
    - SIAM3D Spatial Intersection Attention Module on skips and bottleneck
    - Method 1: Inter-Level Multi-Stage Hinge Regularization anchored to d0 (Tr = 0.20, lambda = 1.0)
    - Loss: Standard compound DC + CE Loss
    
    Isolates the architectural gain of SIAM attention on regularized Depthwise representations.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        # Original Method 1 Hyperparameters
        self.Tr = 0.20
        self.inter_loss_weight = 1.0
        self.stage_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

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

