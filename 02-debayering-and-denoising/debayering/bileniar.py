from scipy.ndimage.filters import convolve

from colour.hints import ArrayLike, Literal, NDArray
from colour.utilities import as_float_array, tstack

from .utils import masks_CFA_Bayer


def bayer2rgb(
    CFA: ArrayLike,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"],
) -> NDArray:
    """
    Return the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    bilinear interpolation.
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
    :cite:`Losson, O. - Comparison of Color Demosaicing Methods`
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
    array([[[ 0.07745098,  0.17941177,  0.84705885],
            [ 0.15490197,  0.4509804 ,  0.5882353 ],
            [ 0.15196079,  0.27450981,  0.61176471],
            [ 0.22352942,  0.5647059 ,  0.30588235]],
           [[ 0.23235295,  0.53529412,  0.28235295],
            [ 0.4647059 ,  0.26960785,  0.19607843],
            [ 0.45588237,  0.4509804 ,  0.20392157],
            [ 0.67058827,  0.18431373,  0.10196078]]])
    """

    CFA = as_float_array(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    H_G = (
        as_float_array(
            [
                [0, 1, 0],
                [1, 4, 1],
                [0, 1, 0],
            ]
        )
        / 4
    )

    H_RB = (
        as_float_array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]
        )
        / 4
    )

    R = convolve(CFA * R_m, H_RB)
    G = convolve(CFA * G_m, H_G)
    B = convolve(CFA * B_m, H_RB)

    del R_m, G_m, B_m, H_RB, H_G

    return tstack([R, G, B])