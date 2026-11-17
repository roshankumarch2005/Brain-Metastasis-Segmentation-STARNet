import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context


class Decoder1HalfResWrapper(nn.Module):
    """
    Wraps standard PlainConvUNet to take the PRIMARY segmentation directly
    from Decoder 1 (d_1 at H/2 x W/2 x D/2), completely bypassing Decoder 0.
    
    Decoder 1 features have double the receptive field and richer semantic pooling,
    reducing false-positive noise and single-voxel edge ambiguity.
    
    During training (deep supervision):
      output[0] = d_1 upsampled to full resolution (H x W x D) via trilinear interpolation
      output[1] = d_1 native half resolution (H/2 x W/2 x D/2)
      output[2] = d_2 quarter resolution (H/4 x W/4 x D/4)
      output[3] = d_3 eighth resolution (H/8 x W/8 x D/8)
      output[4] = d_4 sixteenth resolution (H/16 x W/16 x D/16)
      
    During inference / validation:
      Returns d_1 upsampled directly to the full-resolution scan dimensions.
    """
    def __init__(self, original_net):
        super().__init__()
        self.original_net = original_net

    @property
    def decoder(self):
        return self.original_net.decoder

    @property
    def encoder(self):
        return self.original_net.encoder

    def forward(self, x):
        skips = self.original_net.encoder(x)
        
        # Always obtain full decoder stage representations:
        prev_ds = self.original_net.decoder.deep_supervision
        self.original_net.decoder.deep_supervision = True
        seg_outputs = self.original_net.decoder(skips)
        self.original_net.decoder.deep_supervision = prev_ds

        # seg_outputs[1] is Decoder 1: shape (B, num_classes, H/2, W/2, D/2)
        d1 = seg_outputs[1]
        d1_up = F.interpolate(d1, size=x.shape[2:], mode='trilinear', align_corners=False)

        if prev_ds:
            # Deep supervision: output[0] is d1_up; output[1..4] are [d1, d2, d3, d4]
            # Decoder 0 (seg_outputs[0]) is completely bypassed!
            return [d1_up, seg_outputs[1], seg_outputs[2], seg_outputs[3], seg_outputs[4]]
        else:
            # Inference: segmentation is determined 100% by Decoder 1
            return d1_up


class nnUNetTrainer_Decoder1_HalfRes(nnUNetTrainer):
    """
    Method 2: Primary Segmentation from Decoder 1 (H/2 x W/2) with 3D Max-Pooled Labels.
    
    Addresses the small object vanishing problem during downsampling:
    1. Rather than nearest-neighbor downsampling (which erases micro-lesions < 5,000 voxels),
       labels are downsampled using 3D Max-Pooling (kernel_size=2, stride=2).
       If ANY voxel in a 2x2x2 cube is positive, the downsampled label is 1.
       Zero micro-lesions are lost!
    2. Primary segmentation is drawn directly from Decoder 1 (H/2 x W/2 x D/2).
    3. Trilinear interpolation upsamples Decoder 1 logits to full resolution for
       direct, seamless evaluation against original ground truth.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        maybe_mkdir_p(self.output_folder)

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
        wrapped_network = Decoder1HalfResWrapper(network)
        return wrapped_network

    def get_maxpool_target_pyramid(self, target_0):
        """
        Creates a 5-level multi-scale ground truth pyramid using 3D Max-Pooling:
        - target_0: Full resolution (80, 192, 160)
        - target_1: Half resolution (40, 96, 80) via 3D MaxPool (kernel=2, stride=2)
        - target_2: Quarter resolution (20, 48, 40) via 3D MaxPool
        - target_3: Eighth resolution (10, 24, 20) via 3D MaxPool
        - target_4: Sixteenth resolution (5, 12, 10) via 3D MaxPool
        
        Guarantees micro-lesions are 100% preserved at all scales.
        """
        t0 = target_0.float()
        t1 = F.max_pool3d(t0, kernel_size=2, stride=2)
        t2 = F.max_pool3d(t1, kernel_size=2, stride=2)
        t3 = F.max_pool3d(t2, kernel_size=2, stride=2)
        t4 = F.max_pool3d(t3, kernel_size=2, stride=2)
        return [
            target_0,
            t1.to(target_0.dtype),
            t2.to(target_0.dtype),
            t3.to(target_0.dtype),
            t4.to(target_0.dtype)
        ]

    def get_targets(self, batch: dict):
        if len(self.label_manager.all_labels) <= 2:
            raw_target_0 = batch['target'][0].to(self.device, non_blocking=True)
            return self.get_maxpool_target_pyramid(raw_target_0), raw_target_0
        else:
            targets = [t.to(self.device, non_blocking=True) for t in batch['target'][:5]]
            return targets, targets[0]

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        targets, raw_target_0 = self.get_targets(batch)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast():
            output = self.network(data)
            l = self.loss(output, targets)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': float(l.detach().cpu().numpy())}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        targets, raw_target_0 = self.get_targets(batch)

        with autocast():
            output = self.network(data)
            del data
            l = self.loss(output, targets)

        # Online evaluation on primary prediction: output[0] (Decoder 1 upsampled) vs raw_target_0
        output = output[0]
        target = raw_target_0

        axes = [0] + list(range(2, len(output.shape)))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        mask = None
        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': float(l.detach().cpu().numpy()),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard
        }
