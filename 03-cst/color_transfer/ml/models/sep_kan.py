import torch
from torch import nn
from typing import List
from ..layers import SepKANLayer
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np


class SepKanModel(nn.Module):
    """
    Standard architecture for Kolmogorov-Arnold Networks described in the original paper.
    Layers are defined via a list of layer widths.

    old kan.Model class
    """

    def __init__(
        self,
        in_dims: List[int],
        out_dims: List[int],
        grid_size: int,
        spline_order: int,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ) -> None:
        super(SepKanModel, self).__init__()

        kan_size = [s for s in zip(in_dims, out_dims)]

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.layers = []
        for in_dim, out_dim in kan_size:
            self.layers.append(
                SepKANLayer(in_dim=in_dim,
                         out_dim=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range))

        # Arbitrary layers configuration fc
        self.layers_num = len(self.layers)
        self.kan_params_num = 0
        self.kan_params_indices = [0]

        for layer in self.layers:
            coef_len = np.prod(layer.activation_fn.coef_shape)
            univariate_weight_len = np.prod(layer.residual_layer.univariate_weight_shape)
            residual_weight_len = np.prod(layer.residual_layer.residual_weight_shape)
            self.kan_params_indices.extend([coef_len, univariate_weight_len, residual_weight_len])

        self.kan_params_num = np.sum(self.kan_params_indices)
        self.kan_params_indices = np.cumsum(self.kan_params_indices)

        self.fc = torch.nn.Linear(1000, self.kan_params_num)

    def kan(self, x, w):

        z = []
        for b in range(x.shape[0]):
            y = x[b]
            for n in range(self.layers_num):
                layer = self.layers[n]
                i,j = self.kan_params_indices[3 * n + 0], self.kan_params_indices[3 * n + 1]
                coef = w[b, i:j].view(*layer.activation_fn.coef_shape)
                i,j = self.kan_params_indices[3 * n + 1], self.kan_params_indices[3 * n + 2]
                univariate_weight = w[b, i:j].view(*layer.residual_layer.univariate_weight_shape)
                i,j = self.kan_params_indices[3 * n + 2], self.kan_params_indices[3 * n + 3]
                residual_weight = w[b, i:j].view(*layer.residual_layer.residual_weight_shape)
                y = layer(y, coef, univariate_weight, residual_weight)
            z.append(y)

        return torch.concat(z, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x

        B, C, H, W = img.shape

        # lut (b, c, m)
        w = self.resnet(img)
        w = self.fc(w).view(B, self.kan_params_num)

        # img (b, 3, h, w)
        img = img.permute(0, 2, 3, 1)
        img = img.view(B, H * W, C)

        out = self.kan(img, w)

        out = out.view(B, H, W, C)
        out = out.permute(0, 3, 1, 2)

        return out    
