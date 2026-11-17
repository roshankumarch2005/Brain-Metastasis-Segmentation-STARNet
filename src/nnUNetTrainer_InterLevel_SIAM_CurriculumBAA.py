import os
import torch
import torch.nn as nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.losses.baa_loss import SmoothedBAAClassBalanced_DC_and_CE_loss
import numpy as np

from .nnUNetTrainer_InterLevel_SIAM import nnUNetTrainer_InterLevel_SIAM


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
                        f"\n{'='*75}\n"
                        f"🎓 CURRICULUM LOSS SWITCH ACTIVATED at Epoch {self.trainer.current_epoch}!\n"
                        f"Transitioning from Stage 1 (Default DC+CE) to Stage 2 (Area-Aware BAA Sensitive, beta=1.5).\n"
                        f"{'='*75}\n"
                    )
                self.has_logged_switch = True
            return self.loss_stage2(*args)


class nnUNetTrainer_InterLevel_SIAM_CurriculumBAA(nnUNetTrainer_InterLevel_SIAM):
    """
    Two-Stage Curriculum Learning Trainer with SIAM Attention and Method 1 Multi-Stage Regularization:
    - Architecture: SIAM Attention on skips & bottleneck + 6-stage Inter-Level Discrepancy Head
    - Loss Stage 1 (Epochs 0 to 499): Default nnU-Net DC + CE Loss
    - Loss Stage 2 (Epochs 500 to 1000): Area-Aware BAA Sensitive Loss (beta=1.5)
    - Regularization (Epochs 0 to 1000): Quadratic hinge discrepancy anchored to d0 (Tr = 0.20)
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

