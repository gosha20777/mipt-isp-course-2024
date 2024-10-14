"""
Bayer CFA Masks
===============
*Bayer* CFA (Colour Filter Array) masks generation.
"""

from __future__ import annotations

import numpy as np

from colour.hints import Literal, NDArray, Tuple
from colour.utilities import validate_method
import imageio
import cv2
import os
import tensorflow as tf
import numpy as np
import colour


def read_target_image(path: str, size):
    image = colour.io.read_image(path)[:,:,:3]
    return image.astype(np.float32)


def read_bayer_image(path: str):
    raw = np.asarray(imageio.imread(path))
    if raw is None:
        raise Exception(f'Can not read image {path}')
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    return combined.astype(np.float32) / (4 * 255) / 64


def resize_bayer_image(image, w, h, color=(0, 0, 0, 0)):
    """Create new image(numpy array) filled with certain color in BGR"""
    r_image = np.zeros((h, w, 4), np.float32)
    r_image[:] = color
    r_image[:image.shape[0],:image.shape[1],:image.shape[2]] = image[:min(image.shape[0], w),:min(image.shape[1], h),:image.shape[2]]
    return r_image


def masks_CFA_Bayer(
    shape: int | Tuple[int, ...],
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] | str = "RGGB",
) -> Tuple[NDArray, ...]:
    """
    Return the *Bayer* CFA red, green and blue masks for given pattern.
    Parameters
    ----------
    shape
        Dimensions of the *Bayer* CFA.
    pattern
        Arrangement of the colour filters on the pixel array.
    Returns
    -------
    :class:`tuple`
        *Bayer* CFA red, green and blue masks.
    Examples
    --------
    >>> from pprint import pprint
    >>> shape = (3, 3)
    >>> pprint(masks_CFA_Bayer(shape))
    (array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool))
    >>> pprint(masks_CFA_Bayer(shape, "BGGR"))
    (array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool))
    """


    #pattern = validate_method(
    #    pattern,
    #    ["RGGB", "BGGR", "GRBG", "GBRG"],
    #    '"{0}" CFA pattern is invalid, it must be one of {1}!',
    #).upper()

    channels = {channel: np.zeros(shape, dtype="bool") for channel in "RGB"}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels.values())


def evaluate(method, path):
    train_input_dir = os.path.join(path, "input")
    train_target_dir = os.path.join(path, "groundtruth")


    input_img_paths = sorted(
        [
            os.path.join(train_input_dir, fname)
            for fname in os.listdir(train_input_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(train_target_dir, fname)
            for fname in os.listdir(train_target_dir)
            if fname.endswith(".png")
        ]
    )

    psnr = 0.0
    ssim = 0.0

    for i in range(len(input_img_paths)):
        raw = colour.io.read_image(input_img_paths[i])
        rgb = method.bayer2rgb(raw, pattern="RGGB")
        gt = colour.io.read_image(target_img_paths[i])[:,:,:3].astype(np.float32)
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        gt = tf.convert_to_tensor(gt)
        rgb = tf.convert_to_tensor(rgb)

        psnr = psnr + tf.image.psnr(rgb, gt, max_val=1.0).numpy()
        ssim = ssim + tf.image.ssim(rgb, gt, max_val=1.0).numpy()
    return psnr / len(input_img_paths), ssim / len(input_img_paths)
