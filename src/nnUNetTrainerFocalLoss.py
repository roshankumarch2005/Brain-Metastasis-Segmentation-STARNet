import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_Focal_loss

class nnUNetTrainerFocalLoss(nnUNetTrainer):
    def build_loss(self):
        loss = DC_and_Focal_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 
             'do_bg': False, 
             'ddp': self.is_ddp},
            {'alpha': 0.25, 
             'gamma': 2.0, 
             'apply_nonlin': torch.nn.Softmax(dim=1)}
        )
        return loss