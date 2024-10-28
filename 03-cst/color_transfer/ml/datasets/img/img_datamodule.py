import os
import random
import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .img_dataset import Image2ImageDataset
from color_transfer.ml.transforms.pair_trransform import PairTransform
import numpy as np


class ImgDataModule(L.LightningDataModule):
    def __init__(
            self,
            x_img: np.ndarray,
            y_img: np.ndarray,
            batch_size: int = 1,
            val_batch_size: int = 1,
            test_batch_size: int = 1,
            num_workers: int = min(12, os.cpu_count() - 1),
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None

        random.seed(seed)
        self.x_img = x_img
        self.y_img = y_img
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.image_p_transform = PairTransform(
            crop_size=128, p=0.5, seed=seed
        )
        self.val_image_p_transform = PairTransform(
            crop_size=128, p=0.0, seed=seed
        )

        self.image_train_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_val_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_test_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Image2ImageDataset(
                self.x_img, self.y_img, self.image_train_transform, self.image_p_transform,
            )
            self.val_dataset = Image2ImageDataset(
                self.x_img, self.y_img, self.image_val_transform, self.val_image_p_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Image2ImageDataset(
                self.x_img, self.y_img, self.image_test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
