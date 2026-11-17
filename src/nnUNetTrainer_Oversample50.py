from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_Oversample50(nnUNetTrainer):
    # We do NOT override __init__, completely avoiding all version-specific signature errors.
    
    def initialize(self):
        # 1. Overwrite the default 0.33 set by the base class
        self.oversample_foreground_percent = 0.50
        
        # 2. Proceed with the standard initialization (building data loaders) using the new value
        super().initialize()