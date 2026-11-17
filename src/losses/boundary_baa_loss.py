import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryAware_BAA_Loss(nn.Module):
    """
    Advanced Distance-Aware BAA Loss.
    Calculates the 3D Euclidean Distance Transform (EDT) of the ground truth boundaries.
    The Cross-Entropy penalty is scaled with the physical distance from the true boundary.
    """
    def __init__(self, class_freqs=None, beta_cb=0.999, boundary_scale=1.5, 
                 dice_kwargs=None, ce_kwargs=None, weight_ce=1.0, weight_dice=1.0, ignore_label=None):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.boundary_scale = boundary_scale
        self.beta_cb = beta_cb
        
        # 1. Class Balanced Weights (Cui et al.) - ONLY for Multi-Class (C > 2)
        # For Binary (C <= 2), background is 99% of volume; downweighting background to 0.017 causes collapse.
        if class_freqs is not None and len(class_freqs) > 2:
            class_freq = torch.tensor(class_freqs, dtype=torch.float32)
            eff_num = 1.0 - torch.pow(beta_cb, class_freq)
            alpha_c = (1.0 - beta_cb) / (eff_num + 1e-6)
            alpha_c = alpha_c / alpha_c.sum() * len(class_freqs)
            self.register_buffer("alpha_c", alpha_c)
            self.ce = nn.CrossEntropyLoss(weight=self.alpha_c, reduction='none', **(ce_kwargs or {}))
        else:
            self.alpha_c = None
            self.ce = nn.CrossEntropyLoss(reduction='none', **(ce_kwargs or {}))
            
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **(dice_kwargs or {}))

    def compute_edts_for_batch(self, target_np, num_classes):
        """
        Calculates the Euclidean Distance Transform (EDT) on the CPU.
        target_np shape: (B, 1, D, H, W)
        """
        B, _, D, H, W = target_np.shape
        dist_maps = np.zeros((B, num_classes, D, H, W), dtype=np.float32)
        
        for b in range(B):
            for c in range(1, num_classes): # Skip background (class 0)
                posmask = (target_np[b, 0] == c)
                if not posmask.any():
                    continue
                negmask = ~posmask
                dist_maps[b, c] = distance_transform_edt(negmask)
        return dist_maps

    def forward(self, net_output, target):
        """
        net_output: (B, C, D, H, W)
        target: (B, 1, D, H, W)
        """
        B, C, D, H, W = net_output.shape
        
        # Ensure alpha_c matches C for multi-class
        if C > 2:
            if self.alpha_c is not None and self.alpha_c.shape[0] != C:
                if C == 3:
                    freq = torch.tensor([0.950, 0.045, 0.005], dtype=torch.float32, device=net_output.device)
                elif C == 4:
                    freq = torch.tensor([0.980, 0.005, 0.010, 0.005], dtype=torch.float32, device=net_output.device)
                else:
                    freq = torch.ones(C, dtype=torch.float32, device=net_output.device) / C
                eff_num = 1.0 - torch.pow(self.beta_cb, freq)
                alpha_c = (1.0 - self.beta_cb) / (eff_num + 1e-6)
                alpha_c = alpha_c / alpha_c.sum() * C
                self.alpha_c = alpha_c
                self.ce = nn.CrossEntropyLoss(weight=self.alpha_c, reduction='none')
        else:
            if not isinstance(self.ce, nn.CrossEntropyLoss) or self.ce.weight is not None:
                self.ce = nn.CrossEntropyLoss(reduction='none')

        # 1. Base Cross Entropy
        if self.ignore_label is not None:
            mask = (target != self.ignore_label)
            target_ce = target.clone()
            target_ce[~mask] = 0
            ce_loss = self.ce(net_output, target_ce[:, 0].long())
            ce_loss = ce_loss * mask[:, 0].float()
        else:
            ce_loss = self.ce(net_output, target[:, 0].long()) # (B, D, H, W)
            
        # 2. Boundary Distance Penalty (Explicit HD95 Optimization)
        with torch.no_grad():
            target_np = target.cpu().numpy()
            dist_maps_np = self.compute_edts_for_batch(target_np, C)
            dist_maps_tensor = torch.from_numpy(dist_maps_np).to(net_output.device)
            dist_penalty = 1.0 + self.boundary_scale * torch.log1p(dist_maps_tensor) # (B, C, D, H, W)
            pred_classes = torch.argmax(net_output, dim=1, keepdim=True) # (B, 1, D, H, W)
            spatial_penalty = torch.gather(dist_penalty, 1, pred_classes).squeeze(1) # (B, D, H, W)

        # 3. Apply Boundary Distance Multiplier to Cross Entropy
        ce_loss = (ce_loss * spatial_penalty).mean()
        
        # 4. Standard Soft Dice Loss (with softmax_helper_dim1)
        dice_loss = self.dice(net_output, target)
        
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
