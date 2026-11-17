import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class InterLevelUNetWrapper(nn.Module):
    """
    Wraps standard PlainConvUNet to attach a 1x1x1 segmentation head 
    directly onto the deepest raw encoder bottleneck (skips[-1]).
    
    This ensures that when deep supervision is enabled, the forward pass returns
    ALL 6 levels of predictions:
      output[0] = d_0 (Full Resolution 1x - The Anchor)
      output[1] = d_1 (Half Resolution 1/2x)
      output[2] = d_2 (Quarter Resolution 1/4x)
      output[3] = d_3 (Eighth Resolution 1/8x)
      output[4] = d_4 (Sixteenth Resolution 1/16x)
      output[5] = d_5 (Thirty-second Resolution 1/32x - Raw Bottleneck Head)
    """
    def __init__(self, original_net, num_classes=2):
        super().__init__()
        self.original_net = original_net
        # In 3D fullres, bottleneck has 320 channels:
        bottleneck_channels = original_net.encoder.output_channels[-1]
        self.bottleneck_seg_head = nn.Conv3d(bottleneck_channels, num_classes, kernel_size=1)
        
    @property
    def decoder(self):
        return self.original_net.decoder

    @property
    def encoder(self):
        return self.original_net.encoder

    def forward(self, x):
        skips = self.original_net.encoder(x)
        seg_outputs = self.original_net.decoder(skips)
        
        # When deep supervision is enabled during training:
        if isinstance(seg_outputs, (list, tuple)):
            bottleneck_seg = self.bottleneck_seg_head(skips[-1])
            return list(seg_outputs) + [bottleneck_seg]
        
        # During inference (single full-resolution tensor):
        return seg_outputs


class nnUNetTrainer_InterLevelAll(nnUNetTrainer):
    """
    Method 1: Multi-Scale Discrepancy Penalty Anchored to Decoder 0 (d_0).
    
    Addresses the small object vanishing problem in deep layers:
    - Anchors all deeper decoder stages (d_1, d_2, d_3, d_4) and the deepest bottleneck (d_5)
      directly to d_0 (the full-resolution prediction).
    - Upsamples each coarse stage to the spatial dimensions of d_0.
    - Applies quadratic hinge loss: max(0, |Upsample(d_k) - d_0| - Tr)^2.
    - Directly injects high-frequency spatial gradients into deeper layers during backprop.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        maybe_mkdir_p(self.output_folder)
        self.Tr = 0.20                       # Discrepancy threshold (20% probability difference)
        self.inter_loss_weight = 1.0         # Overall weight for inter-level discrepancy loss
        # Stage weights: [d1-d0, d2-d0, d3-d0, d4-d0, d5_bottleneck-d0]
        self.stage_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    def set_deep_supervision_enabled(self, enabled: bool):
        net = self.network.module if self.is_ddp else self.network
        if hasattr(net, 'original_net'):
            net.original_net.decoder.deep_supervision = enabled
        elif hasattr(net, 'decoder'):
            net.decoder.deep_supervision = enabled

    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision: bool = True):
        network = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision
        )
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        wrapped_network = InterLevelUNetWrapper(network, num_classes=num_classes)
        return wrapped_network

    def compute_inter_level_discrepancy(self, outputs):
        """
        Computes the inter-level discrepancy penalty across all 5 deeper stages relative to d_0.
        Uses d_0.detach() as the teacher so gradients only pull the deeper layers towards d_0
        without letting coarse layers corrupt d_0's high-resolution precision.
        """
        # Convert logits to foreground class probabilities via softmax:
        probs = [torch.softmax(out, dim=1)[:, 1:] for out in outputs]
        p0 = probs[0].detach()  # Anchor teacher: shape (B, 1, D, H, W)
        
        loss = torch.tensor(0.0, device=p0.device)
        for k in range(1, len(probs)):
            pk = probs[k]
            # Spatial upsampling to match p0's exact grid dimensions:
            pk_up = F.interpolate(pk, size=p0.shape[2:], mode='trilinear', align_corners=False)
            diff = torch.abs(pk_up - p0)
            
            # Quadratic hinge penalty when difference exceeds Tr:
            penalty = F.relu(diff - self.Tr) ** 2
            w = self.stage_weights[k-1] if k-1 < len(self.stage_weights) else 0.2
            loss = loss + w * penalty.mean()
            
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        
        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            output = self.network(data)
            # 1. Standard nnU-Net deep supervision loss on the first len(target) heads:
            l_nnunet = self.loss(output[:len(target)], target)
            
            # 2. Professor's inter-level discrepancy penalty across all 5 deeper stages:
            l_inter = self.compute_inter_level_discrepancy(output)
            
            # Combined Loss:
            total_loss = l_nnunet + self.inter_loss_weight * l_inter
            
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return {
            'loss': float(total_loss.detach().cpu().numpy()),
            'l_nnunet': float(l_nnunet.detach().cpu().numpy()),
            'l_inter': float(l_inter.detach().cpu().numpy())
        }
