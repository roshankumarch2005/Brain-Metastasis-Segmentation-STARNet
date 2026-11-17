import torch
import torch.nn as nn

class SIAM2D(nn.Module):
    """
    Parameter-free Spatial Intersection Attention Module (2D)
    Paper: Han et al., Pattern Recognition 2024
    """
    def __init__(self, scale_factors=True):
        super().__init__()
        self.scale_factors = scale_factors

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Step 1: Channel Average Pooling -> (B, 1, H, W)
        z = torch.mean(x, dim=1, keepdim=True)

        # Step 2: Directional 1D mean queries
        A_col = torch.mean(z, dim=2, keepdim=True)  # (B, 1, 1, W)
        A_row = torch.mean(z, dim=3, keepdim=True)  # (B, 1, H, 1)

        lambda_col = W if self.scale_factors else 1.0
        lambda_row = H if self.scale_factors else 1.0

        # Step 3: Dot product / Cross-dimension attention
        # m_col: contracts W dimension -> (B, C, H, 1)
        m_col = torch.matmul(x, A_col.transpose(-1, -2)) / lambda_col
        # m_row: contracts H dimension -> (B, C, 1, W)
        m_row = torch.matmul(x.transpose(-1, -2), A_row).transpose(-1, -2) / lambda_row

        # Step 4: Spatial intersection with ReLU1 clipping
        att = torch.clamp(m_col, 0.0, 1.0) + torch.clamp(m_row, 0.0, 1.0)

        # Additive / Residual recalibration
        return x + x * att


class SIAM3D(nn.Module):
    """
    Volumetric Extension of SIAM for 3D Medical Volumes (B, C, D, H, W)
    """
    def __init__(self, scale_factors=True):
        super().__init__()
        self.scale_factors = scale_factors

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # Step 1: Channel average pooling -> (B, 1, D, H, W)
        z = torch.mean(x, dim=1, keepdim=True)

        # Step 2: Directional queries along D, H, W
        A_d = torch.mean(z, dim=(3, 4), keepdim=True)  # (B, 1, D, 1, 1)
        A_h = torch.mean(z, dim=(2, 4), keepdim=True)  # (B, 1, 1, H, 1)
        A_w = torch.mean(z, dim=(2, 3), keepdim=True)  # (B, 1, 1, 1, W)

        scale_d = (H * W) if self.scale_factors else 1.0
        scale_h = (D * W) if self.scale_factors else 1.0
        scale_w = (D * H) if self.scale_factors else 1.0

        # Step 3: Cross-dimension projection
        m_d = torch.sum(x * A_d, dim=(3, 4), keepdim=True) / scale_d  # (B, C, D, 1, 1)
        m_h = torch.sum(x * A_h, dim=(2, 4), keepdim=True) / scale_h  # (B, C, 1, H, 1)
        m_w = torch.sum(x * A_w, dim=(2, 3), keepdim=True) / scale_w  # (B, C, 1, 1, W)

        # Step 4: 3D spatial intersection
        att = (
            torch.clamp(m_d, 0.0, 1.0)
            + torch.clamp(m_h, 0.0, 1.0)
            + torch.clamp(m_w, 0.0, 1.0)
        )

        return x + x * att
