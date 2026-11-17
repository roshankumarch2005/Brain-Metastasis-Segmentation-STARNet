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
    ALL levels of predictions including the bottleneck head:
      output[0] = d_0 (Full Resolution 1x - The Anchor)
      ...
      output[-1] = d_bottleneck (Thirty-second Resolution 1/32x - Raw Bottleneck Head)
    """
    def __init__(self, original_net, num_classes=2):
        super().__init__()
        self.original_net = original_net
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


class nnUNetTrainer_InterLevelBottleneckOnly(nnUNetTrainer):
    """
    Method 1 Ablation: Direct Bottleneck-to-d0 Jump ONLY (d_bottleneck -> d_0).
    
    Tests the isolated impact of the direct shortcut between the deepest bottleneck 
    representation (1/32x resolution) and the full-resolution prediction (d_0).
    
    No intermediate stage discrepancies are penalized.
    Penalty: max(0, |Upsample(d_bottleneck) - d_0| - Tr)^2
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        maybe_mkdir_p(self.output_folder)
        self.Tr = 0.20                       # Discrepancy threshold (20% probability difference)
        self.inter_loss_weight = 1.0         # Overall weight for bottleneck discrepancy loss

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

    def compute_bottleneck_discrepancy(self, outputs):
        """
        Computes the discrepancy penalty ONLY between the raw bottleneck output (outputs[-1])
        and d_0 (outputs[0]).
        """
        # Convert logits to foreground class probabilities via softmax:
        p0 = torch.softmax(outputs[0], dim=1)[:, 1:].detach()  # High-resolution anchor teacher
        p_bottleneck = torch.softmax(outputs[-1], dim=1)[:, 1:] # Bottleneck prediction (1/32x)
        
        # Upsample bottleneck directly to d_0's spatial grid:
        p_bottleneck_up = F.interpolate(p_bottleneck, size=p0.shape[2:], mode='trilinear', align_corners=False)
        diff = torch.abs(p_bottleneck_up - p0)
        
        # Quadratic hinge penalty:
        penalty = F.relu(diff - self.Tr) ** 2
        return penalty.mean()

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in batch['target']]
        
        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            output = self.network(data)
            # 1. Standard nnU-Net deep supervision loss on the first len(target) heads:
            l_nnunet = self.loss(output[:len(target)], target)
            
            # 2. Bottleneck-only discrepancy penalty:
            l_bottleneck = self.compute_bottleneck_discrepancy(output)
            
            # Combined Loss:
            total_loss = l_nnunet + self.inter_loss_weight * l_bottleneck
            
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return {
            'loss': float(total_loss.detach().cpu().numpy()),
            'l_nnunet': float(l_nnunet.detach().cpu().numpy()),
            'l_bottleneck': float(l_bottleneck.detach().cpu().numpy())
        }

