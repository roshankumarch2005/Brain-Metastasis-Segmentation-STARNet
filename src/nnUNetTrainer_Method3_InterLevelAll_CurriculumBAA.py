import os
import torch
import torch.nn as nn
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Method3_InterLevelAll import nnUNetTrainer_Method3_InterLevelAll
except ImportError:
    try:
        from .nnUNetTrainer_Method3_InterLevelAll import nnUNetTrainer_Method3_InterLevelAll
    except ImportError:
        from nnUNetTrainer_Method3_InterLevelAll import nnUNetTrainer_Method3_InterLevelAll

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import CurriculumLossWrapper
except ImportError:
    try:
        from .nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import CurriculumLossWrapper
    except ImportError:
        from nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import CurriculumLossWrapper


class nnUNetTrainer_Method3_InterLevelAll_CurriculumBAA(nnUNetTrainer_Method3_InterLevelAll):
    """
    Configuration 14: 3D Depthwise-Separable Architecture + Method 1 Inter-Level Hinge + Curriculum BAA.
    
    Ablation Study: Evaluates a pure loss-driven boundary regularizer (Method 1 Multi-Stage Hinge
    with Tr=0.20, lambda=1.0 AND Two-Stage Curriculum BAA beta=1.5) on the lightweight 1.23M depthwise
    backbone WITHOUT the SIAM spatial attention module.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        maybe_mkdir_p(self.output_folder)
        self.num_epochs = 1000
        # Method 1 Regularization Hyperparameters
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
