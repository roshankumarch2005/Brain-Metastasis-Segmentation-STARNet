import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .models.siam import SIAM2D, SIAM3D

class SIAMEncoderWrapper(nn.Module):
    """
    Wraps the standard nnU-Net encoder to apply SIAM attention 
    at the skip outputs of each encoder stage and at the bottleneck.
    """
    def __init__(self, original_encoder, is_3d: bool):
        super().__init__()
        self.original_encoder = original_encoder
        self.siam = SIAM3D() if is_3d else SIAM2D()

    def forward(self, x):
        skips = self.original_encoder(x)
        
        # Apply SIAM to all skip outputs and bottleneck
        if isinstance(skips, (list, tuple)):
            new_skips = [self.siam(skip) for skip in skips]
            if isinstance(skips, tuple):
                return tuple(new_skips)
            return new_skips
        
        return self.siam(skips)

class nnUNetTrainerSIAM(nnUNetTrainer):
    """
    Custom nnU-Net Trainer that integrates the Spatial Intersection Attention Module (SIAM).
    It intercepts the network creation and wraps the encoder.
    """
    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision: bool = True):
        # Build the standard network (PlainConvUNet or ResidualEncoderUNet)
        network = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision
        )
        
        # Determine if the network is 2D or 3D based on patch size
        is_3d = len(configuration_manager.patch_size) == 3
        
        # Wrap the encoder to inject SIAM attention
        if hasattr(network, 'encoder'):
            network.encoder = SIAMEncoderWrapper(network.encoder, is_3d)
        else:
            print("WARNING: Network does not have an 'encoder' attribute. SIAM not applied.")
            
        return network

    def initialize(self):
        super().initialize()
        if hasattr(self, 'loss') and hasattr(self.loss, 'to'):
            self.loss.to(self.device)

    def _build_loss(self):
        if hasattr(self, 'build_loss') and callable(getattr(self, 'build_loss')):
            return self.build_loss()
        return super()._build_loss()


class nnUNetTrainerSIAM_BAA(nnUNetTrainerSIAM):
    """
    Trainer that uses the SIAM architecture but overrides the loss function 
    to use the custom Balanced Area-Aware (BAA) Class-Balanced Loss.
    """
    def build_loss(self):
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
        import numpy as np

        # Dynamically adapt class frequencies: 2-class binary vs 4-class BraTS
        num_classes = self.label_manager.num_segmentation_heads
        if num_classes == 2:
            class_freqs = [0.992, 0.008]
        elif num_classes == 3:
            class_freqs = [0.950, 0.045, 0.005]
        elif num_classes == 4:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        else:
            class_freqs = [0.98] + [0.02 / (num_classes - 1)] * (num_classes - 1)
        
        loss = SmoothedBAAClassBalanced_DC_and_CE_loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            beta_scale=0.75,
            area_threshold=5000, # Voxel count threshold for "small" object
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        
        return DeepSupervisionWrapper(loss, weights)

class nnUNetTrainerSIAM_BAA_Sensitive(nnUNetTrainerSIAM_BAA):
    """
    A more aggressive version of the BAA trainer, aimed at capturing extremely tiny
    micro-lesions. It doubles the beta_scale to punish misses more violently.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        
    def build_loss(self):
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
        import numpy as np

        # Dynamically adapt class frequencies: 2-class binary vs 4-class BraTS
        num_classes = self.label_manager.num_segmentation_heads
        if num_classes == 2:
            class_freqs = [0.992, 0.008]
        elif num_classes == 3:
            class_freqs = [0.950, 0.045, 0.005]
        elif num_classes == 4:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        else:
            class_freqs = [0.98] + [0.02 / (num_classes - 1)] * (num_classes - 1)
        
        loss = SmoothedBAAClassBalanced_DC_and_CE_loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            beta_scale=1.5, # HYPERSENSITIVE: Increased from 0.75
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
        
        return DeepSupervisionWrapper(loss, weights)

class nnUNetTrainerSIAM_OnlyDice(nnUNetTrainerSIAM_BAA):
    """
    Experimental trainer for ablation study: 
    Trains using ONLY Dice Loss (Cross-Entropy weight = 0.0)
    """
    def build_loss(self):
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
        import numpy as np

        if self.label_manager.num_segmentation_heads == 2:
            class_freqs = [0.992, 0.008]
        elif self.label_manager.num_segmentation_heads == 3:
            class_freqs = [0.950, 0.045, 0.005]
        else:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        
        loss = SmoothedBAAClassBalanced_DC_and_CE_loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            beta_scale=0.0, # Turn off BAA multiplier for baseline test
            area_threshold=5000,
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_ce=0.0,     # ONLY DICE
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        
        return DeepSupervisionWrapper(loss, weights)

class nnUNetTrainerSIAM_OnlyCE(nnUNetTrainerSIAM_BAA):
    """
    Experimental trainer for ablation study: 
    Trains using ONLY Cross-Entropy Loss (Dice weight = 0.0)
    """
    def build_loss(self):
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
        import numpy as np

        if self.label_manager.num_segmentation_heads == 2:
            class_freqs = [0.992, 0.008]
        elif self.label_manager.num_segmentation_heads == 3:
            class_freqs = [0.950, 0.045, 0.005]
        else:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        
        loss = SmoothedBAAClassBalanced_DC_and_CE_loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            beta_scale=0.0, # Turn off BAA multiplier for baseline test
            area_threshold=5000,
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_ce=1.0,     # ONLY CROSS-ENTROPY
            weight_dice=0.0,
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        
        return DeepSupervisionWrapper(loss, weights)

class nnUNetTrainerSIAM_Boundary_BAA(nnUNetTrainerSIAM):
    """
    Trainer that uses the explicit HD95-optimizing Boundary Distance Loss.
    """
    def build_loss(self):
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.boundary_baa_loss import BoundaryAware_BAA_Loss
        import numpy as np

        if self.label_manager.num_segmentation_heads == 2:
            class_freqs = [0.992, 0.008]
        elif self.label_manager.num_segmentation_heads == 3:
            class_freqs = [0.950, 0.045, 0.005]
        else:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        
        loss = BoundaryAware_BAA_Loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            boundary_scale=1.5, # Our new HD95 scaling factor
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        
        return DeepSupervisionWrapper(loss, weights)

class nnUNetTrainerSIAM_Boundary_Alpha08(nnUNetTrainerSIAM_Boundary_BAA):
    """
    Experimental trainer evaluating Convex Weighting of the Boundary Loss.
    Prioritizes volume preservation by assigning 80% weight to Dice Loss 
    and 20% weight to the Cross-Entropy Boundary Penalty.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000

    def build_loss(self):
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.boundary_baa_loss import BoundaryAware_BAA_Loss
        import numpy as np

        if self.label_manager.num_segmentation_heads == 2:
            class_freqs = [0.992, 0.008]
        elif self.label_manager.num_segmentation_heads == 3:
            class_freqs = [0.950, 0.045, 0.005]
        else:
            class_freqs = [0.980, 0.005, 0.010, 0.005]
        
        loss = BoundaryAware_BAA_Loss(
            class_freqs=class_freqs,
            beta_cb=0.999,
            boundary_scale=1.5,
            dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_dice=0.8, # Volume preservation priority
            weight_ce=0.2,   # Boundary penalty
            ignore_label=self.label_manager.ignore_label
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        
        return DeepSupervisionWrapper(loss, weights)


class CurriculumLossWrapper(torch.nn.Module):
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
                        f"\n{'='*75}\n"
                        f"🎓 CURRICULUM LOSS SWITCH ACTIVATED at Epoch {self.trainer.current_epoch}!\n"
                        f"Transitioning from Stage 1 (Default DC+CE) to Stage 2 (Area-Aware BAA Sensitive, beta=1.5).\n"
                        f"{'='*75}\n"
                    )
                self.has_logged_switch = True
            return self.loss_stage2(*args)


class nnUNetTrainerSIAM_CurriculumBAA(nnUNetTrainerSIAM_BAA_Sensitive):
    """
    Two-Stage Curriculum Learning Trainer with SIAM Attention:
    - Stage 1 (Epochs 0 to 499): Default nnU-Net DC + CE Loss
    - Stage 2 (Epochs 500 to 1000): Area-Aware BAA Sensitive Loss (beta=1.5)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000

    def initialize(self):
        super().initialize()
        if hasattr(self, 'loss') and hasattr(self.loss, 'to'):
            self.loss.to(self.device)

    def _build_loss(self):
        return self.build_loss()

    def build_loss(self):
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from .losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
        import numpy as np

        # 1. Build Stage 1 Loss (Default DC + CE)
        loss_default = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # 2. Build Stage 2 Loss (Area-Aware BAA Sensitive Loss beta=1.5)
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

