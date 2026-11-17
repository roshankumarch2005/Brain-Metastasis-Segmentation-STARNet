import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class SmoothedBAAClassBalanced_DC_and_CE_loss(nn.Module):
    """
    Balanced Area-Aware (BAA) Class-Balanced Cross-Entropy and Dice Loss.
    Adapted for 3D MRI Volumetric Segmentation.
    """
    def __init__(self, class_freqs=None, beta_cb=0.999, beta_scale=0.75, area_threshold=5000, 
                 dice_kwargs=None, ce_kwargs=None, weight_ce=1.0, weight_dice=1.0, ignore_label=None):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.beta_cb = beta_cb
        
        # 1. Class Balanced Weights (Cui et al.) - ONLY for Multi-Class (C > 2)
        # In binary segmentation (C <= 2), background occupies 99% of volume; downweighting background
        # to 0.017 destroys the penalty against false positives and causes all-foreground collapse.
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
            
        self.beta_scale = beta_scale
        self.area_threshold = area_threshold
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **(dice_kwargs or {}))
        
    def forward(self, net_output, target):
        """
        net_output: (B, C, D, H, W)
        target: (B, 1, D, H, W)
        """
        B, C, D, H, W = net_output.shape

        # Ensure alpha_c matches C for multi-class
        if C > 2:
            if self.alpha_c is not None:
                if self.alpha_c.shape[0] != C:
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
                elif self.alpha_c.device != net_output.device:
                    self.alpha_c = self.alpha_c.to(net_output.device)
                    self.ce = nn.CrossEntropyLoss(weight=self.alpha_c, reduction='none')
        else:
            # For Binary (C <= 2), strictly keep CE unweighted
            if not isinstance(self.ce, nn.CrossEntropyLoss) or self.ce.weight is not None:
                self.ce = nn.CrossEntropyLoss(reduction='none')

        # 1. Compute Base Cross Entropy Loss
        if self.ignore_label is not None:
            mask = (target != self.ignore_label)
            target_ce = target.clone()
            target_ce[~mask] = 0
            ce_loss = self.ce(net_output, target_ce[:, 0].long())
            ce_loss = ce_loss * mask[:, 0].float()
        else:
            ce_loss = self.ce(net_output, target[:, 0].long()) # Shape: (B, D, H, W)
            
        # 2. Volume-Aware Modulation (Scale Imbalance / BAA)
        with torch.no_grad():
            B, C, D, H, W = net_output.shape
            pred_probs = torch.softmax(net_output.float(), dim=1)
            
            # One-hot encode target to compute exact volumes
            target_clamped = target[:, 0].long().clamp(min=0, max=C-1)
            target_one_hot = F.one_hot(target_clamped, num_classes=C).permute(0, 4, 1, 2, 3).float()
            
            # Calculate 3D Volumes (number of voxels) per class per batch element in float32
            vol_pred = pred_probs.sum(dim=(2, 3, 4)) # (B, C)
            vol_gt = target_one_hot.sum(dim=(2, 3, 4)) # (B, C)
            
            # Delta modulation (tanh log ratio to penalize bad scale predictions)
            log_ratio = torch.abs(torch.log((vol_pred + 1.0) / (vol_gt + 1.0)))
            delta_smooth = 1.0 + self.beta_scale * torch.tanh(log_ratio)
            
            # Only apply multiplier to "small" objects (Scale Imbalance)
            small_medium_mask = (vol_gt < self.area_threshold).float()
            delta = 1.0 + small_medium_mask * (delta_smooth - 1.0) # (B, C)
            
            # Map the delta multiplier back to 3D spatial dimensions based on GT class at each voxel
            delta_spatial = torch.gather(delta, 1, target_clamped.flatten(1)).view(B, D, H, W).to(ce_loss.dtype)
        
        # Apply the Volume-Aware multiplier to the Cross Entropy Loss
        ce_loss = (ce_loss * delta_spatial).mean()
        
        # 3. Standard Soft Dice Loss
        dice_loss = self.dice(net_output, target)
        
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
