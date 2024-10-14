from tensorflow import keras
import numpy as np
from ..utils import read_bayer_image, read_target_image
import colour


class RawLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, 
            batch_size, 
            img_size, 
            dslr_scale, 
            input_img_paths, 
            target_img_paths
        ):
        self.batch_size = batch_size
        self.original_img_size = img_size

        if img_size[0] % 2 != 0:
            img_size = (img_size[0] - 1, img_size[1])
        if img_size[1] % 2 != 0:
            img_size = (img_size[0], img_size[1] - 1)
        self.img_size = img_size
        self.dslr_scale = dslr_scale
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        dslr_size = (self.img_size[0] * self.dslr_scale, self.img_size[1] * self.dslr_scale)
    

        x = np.zeros((self.batch_size,) + self.img_size + (4,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = read_bayer_image(path)[:self.img_size[0],:self.img_size[1],:4]

        y = np.zeros((self.batch_size,) + dslr_size + (3,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            dslr_size_ = (self.original_img_size[0] * self.dslr_scale, self.original_img_size[1] * self.dslr_scale)
            img = read_target_image(path, dslr_size_)
            y[j] = img[:dslr_size[0],:dslr_size[1],:3]
        
        return x, y


class RawLoaderFlat(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, 
            batch_size, 
            img_size, 
            input_img_paths, 
            target_img_paths
        ):
        self.batch_size = batch_size

        if img_size[0] % 2 != 0:
            img_size = (img_size[0] - 1, img_size[1])
        if img_size[1] % 2 != 0:
            img_size = (img_size[0], img_size[1] - 1)
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
    

        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = colour.io.read_image(path)[:self.img_size[0],:self.img_size[1]]

        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = colour.io.read_image(path)[:,:,:3].astype(np.float32)
            y[j] = img[:self.img_size[0],:self.img_size[1],:]
        
        return x, y
