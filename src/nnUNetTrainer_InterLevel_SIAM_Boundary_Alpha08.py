import os
import torch
import torch.nn as nn
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.losses.boundary_baa_loss import BoundaryAware_BAA_Loss
import numpy as np

# Import base class
from .nnUNetTrainer_InterLevel_SIAM import nnUNetTrainer_InterLevel_SIAM


class nnUNetTrainer_InterLevel_SIAM_Boundary_Alpha08(nnUNetTrainer_InterLevel_SIAM):
    """
    State-of-the-Art Binary Model:
    Unites Method 1 (Multi-Stage Inter-Level Discrepancy Regularization) + 
           SIAM Attention + 
           Convex 80/20 Boundary-Aware Loss (Our best standalone distance record: 5.58 mm).
    
    1. Architecture: PlainConvUNet + SIAM3D skip attention + Bottleneck 1x1x1 segmentation head
    2. Deep Supervision Loss: BoundaryAware_BAA_Loss with Convex 80% Dice / 20% CE (Boundary scale = 1.5)
    3. Inter-level Regularization: Quadratic hinge discrepancy penalty anchored to d_0 across all 5 deeper levels
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

