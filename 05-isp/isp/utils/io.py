import imageio.v3 as imageio
import numpy as np
from scipy.ndimage.filters import convolve
from colour.hints import ArrayLike, Literal, NDArray, Tuple
from colour.utilities import as_float_array, tstack
import numpy as np


def read_bayer_image(path: str):
    raw = imageio.imread(path)
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    return np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))

def debayer_image(raw: np.ndarray):
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    return np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))


def read_rgb_image(path: str) -> np.ndarray:
    return imageio.imread(path)


def read_numpy_feature(path: str) -> np.ndarray:
    return np.load(path)


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


def bayer2rgb(
    CFA: ArrayLike,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"],
) -> NDArray:
    """
    Return the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    *Malvar (2004)* demosaicing algorithm.
    Parameters
    ----------
    CFA
        *Bayer* CFA.
    pattern
        Arrangement of the colour filters on the pixel array.
    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.
    References
    ----------
    :cite:`Malvar - High-Quality Linear Interpolation for Demosaicing of
    Bayer-Patterned Color Images`
    Examples
    --------
    >>> import numpy as np
    >>> CFA = np.array(
    ...     [
    ...         [0.3764706, 0.360784320, 0.40784314, 0.3764706],
    ...         [0.35686275, 0.30980393, 0.36078432, 0.29803923],
    ...     ]
    ... )
    >>> bayer2rgb(CFA, "BGGR")
    array([[[ 0.35539217,  0.37058825,  0.3764706 ],
            [ 0.34264707,  0.36078432,  0.37450981],
            [ 0.36568628,  0.39607844,  0.40784314],
            [ 0.36568629,  0.3764706 ,  0.3882353 ]],
           [[ 0.34411765,  0.35686275,  0.36200981],
            [ 0.30980393,  0.32990197,  0.34975491],
            [ 0.33039216,  0.36078432,  0.38063726],
            [ 0.29803923,  0.30441178,  0.31740197]]])
    """

    CFA = as_float_array(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    GR_GB = (
        as_float_array(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [-1.0, 2.0, 4.0, 2.0, -1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ]
        )
        / 8
    )

    Rg_RB_Bg_BR = (
        as_float_array(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, -1.0, 0.0, -1.0, 0.0],
                [-1.0, 4.0, 5.0, 4.0, -1.0],
                [0.0, -1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
            ]
        )
        / 8
    )

    Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

    Rb_BB_Br_RR = (
        as_float_array(
            [
                [0.0, 0.0, -1.5, 0.0, 0.0],
                [0.0, 2.0, 0.0, 2.0, 0.0],
                [-1.5, 0.0, 6.0, 0.0, -1.5],
                [0.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, -1.5, 0.0, 0.0],
            ]
        )
        / 8
    )

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    del G_m

    G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve(CFA, GR_GB), G)

    RBg_RBBR = convolve(CFA, Rg_RB_Bg_BR)
    RBg_BRRB = convolve(CFA, Rg_BR_Bg_RB)
    RBgr_BBRR = convolve(CFA, Rb_BB_Br_RR)

    del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    # Red columns.
    R_c = np.any(R_m == 1, axis=0)[None] * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
    # Blue columns
    B_c = np.any(B_m == 1, axis=0)[None] * np.ones(B.shape)

    del R_m, B_m

    R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
    R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

    B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
    B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

    R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
    B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

    del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

    return tstack([R, G, B])