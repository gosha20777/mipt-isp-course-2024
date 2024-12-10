import torch
from torch import nn
from typing import List


class DhIspModel(nn.Module):
    """
    Standard architecture for Kolmogorov-Arnold Networks described in the original paper.
    Layers are defined via a list of layer widths.

    old kan.Model class
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
    ) -> None:
        super(DhIspModel, self).__init__()
        self.model = nn.Sequential(
           nn.Conv2d(in_dim, 16, 3, padding='same'),
           nn.Tanh(),
           nn.Conv2d(16, 16, 3, padding='same'),
           nn.ReLU(inplace=True),
           nn.Conv2d(16, out_dim*4, 3, padding='same'),
           nn.ReLU(inplace=True),
           nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
    

class DhIspModel3ch(nn.Module):
    """
    Standard architecture for Kolmogorov-Arnold Networks described in the original paper.
    Layers are defined via a list of layer widths.

    old kan.Model class
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
    ) -> None:
        super(DhIspModel3ch, self).__init__()
        self.model = nn.Sequential(
           nn.Conv2d(in_dim, 16, 3, padding='same'),
           nn.Tanh(),
           nn.Conv2d(16, 16, 3, padding='same'),
           nn.ReLU(inplace=True),
           nn.Conv2d(16, 12, 3, padding='same'),
           nn.ReLU(inplace=True),
           nn.Conv2d(12, out_dim, 3, padding='same'),
           nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

