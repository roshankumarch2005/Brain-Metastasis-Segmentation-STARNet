import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_InterLevel_SIAM_CurriculumBAA import nnUNetTrainer_InterLevel_SIAM_CurriculumBAA

class nnUNetTrainer_M10_Lam05_Tr20(nnUNetTrainer_InterLevel_SIAM_CurriculumBAA):
    """
    M10 Flagship Hyperparameter Sweep 1:
    - lambda = 0.50 (Weaker Inter-Level Regularization)
    - Tr = 0.20 (Standard Tolerance Threshold)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.inter_loss_weight = 0.50
        self.Tr = 0.20
        self.print_to_log_file(f"🎯 [SWEEP ACTIVE] Model M10 Sweep: lambda={self.inter_loss_weight}, Tr={self.Tr}")


class nnUNetTrainer_M10_Lam15_Tr20(nnUNetTrainer_InterLevel_SIAM_CurriculumBAA):
    """
    M10 Flagship Hyperparameter Sweep 2:
    - lambda = 1.50 (Stronger Inter-Level Regularization)
    - Tr = 0.20 (Standard Tolerance Threshold)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.inter_loss_weight = 1.50
        self.Tr = 0.20
        self.print_to_log_file(f"🎯 [SWEEP ACTIVE] Model M10 Sweep: lambda={self.inter_loss_weight}, Tr={self.Tr}")


class nnUNetTrainer_M10_Lam10_Tr10(nnUNetTrainer_InterLevel_SIAM_CurriculumBAA):
    """
    M10 Flagship Hyperparameter Sweep 3:
    - lambda = 1.00 (Standard Regularization Weight)
    - Tr = 0.10 (Tighter Tolerance Threshold - 10% margin)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.inter_loss_weight = 1.00
        self.Tr = 0.10
        self.print_to_log_file(f"🎯 [SWEEP ACTIVE] Model M10 Sweep: lambda={self.inter_loss_weight}, Tr={self.Tr}")


class nnUNetTrainer_M10_Lam10_Tr30(nnUNetTrainer_InterLevel_SIAM_CurriculumBAA):
    """
    M10 Flagship Hyperparameter Sweep 4:
    - lambda = 1.00 (Standard Regularization Weight)
    - Tr = 0.30 (Looser Tolerance Threshold - 30% margin)
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.inter_loss_weight = 1.00
        self.Tr = 0.30
        self.print_to_log_file(f"🎯 [SWEEP ACTIVE] Model M10 Sweep: lambda={self.inter_loss_weight}, Tr={self.Tr}")
