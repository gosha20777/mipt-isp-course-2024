import torch
import torch.nn as nn
from .conv_sep_kan_layer import ConvSepKANLayer


class ConvSepKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dims, out_dims, kernel_sizes, grid_size, spline_order, residual_std, grid_range):
        super(ConvSepKAN, self).__init__()

        kan_size = [s for s in zip(in_dims, out_dims, kernel_sizes)]

        self.layers = []
        for in_dim, out_dim, kernel_size in kan_size:
            self.layers.append(
                ConvSepKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         kernel_size=kernel_size,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x    
