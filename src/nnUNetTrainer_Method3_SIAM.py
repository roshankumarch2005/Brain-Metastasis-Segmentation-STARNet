import os
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.models.siam import SIAM3D, SIAM2D

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D
except ImportError:
    try:
        from .nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D
    except ImportError:
        from nnUNetTrainer_DepthwiseSeparable3D import DepthwiseSeparableConv3d, nnUNetTrainer_DepthwiseSeparable3D

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_InterLevel_SIAM import SIAMEncoderWrapper
except ImportError:
    try:
        from .nnUNetTrainer_InterLevel_SIAM import SIAMEncoderWrapper
    except ImportError:
        from nnUNetTrainer_InterLevel_SIAM import SIAMEncoderWrapper


class nnUNetTrainer_Method3_SIAM(nnUNetTrainer_DepthwiseSeparable3D):
    """
    Configuration 9: 3D Depthwise-Separable Convolution + Spatial Intersection Attention (SIAM3D).
    
    Ablation Study: Evaluates the isolated architectural benefit of SIAM boundary attention
    on the lightweight depthwise-separable backbone (~1.49M parameters) using standard DC + CE loss
    (without BAA loss, without Method 1 hinge).
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision: bool = True):
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        
        # Base 16, Cap 256 feature progression (~1.49M params with SIAM)
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
        
        # Inject SIAM attention into encoder skips and bottleneck
        is_3d = len(configuration_manager.patch_size) == 3
        if hasattr(network, 'encoder'):
            network.encoder = SIAMEncoderWrapper(network.encoder, is_3d=is_3d)
            
        return network
