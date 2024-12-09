import torch
from torch.utils.data import Dataset
from color_transfer.ml.utils.io import read_rgb_image
from typing import List
from torchvision.transforms.v2 import Compose
from color_transfer.ml.transforms.pair_trransform import PairTransform
import numpy as np


class Image2ImageDataset(Dataset):
    def __init__(self, x_img: np.ndarray, y_img: np.ndarray, transform: Compose, p_transform: PairTransform = None, lengh = 100) -> None:
        self.x_img = x_img
        self.y_img = y_img
        self.lengh = lengh
        self.transform = transform
        self.p_transform = p_transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is not None:
            x = self.transform(self.x_img)

        if self.transform is not None:
            y = self.transform(self.y_img)

        if self.p_transform is not None:
            x, y = self.p_transform(x, y)

        return x, y

    def __len__(self) -> int:
        return self.lengh