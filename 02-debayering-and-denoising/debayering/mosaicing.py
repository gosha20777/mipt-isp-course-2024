from colour.hints import ArrayLike, Literal, NDArray
from colour.utilities import as_float_array, tsplit

from .utils import masks_CFA_Bayer


def rgb2bayer(
    RGB: ArrayLike,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"],
) -> NDArray:
    """
    Return the *Bayer* CFA mosaic for a given *RGB* colourspace array.
    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    pattern
        Arrangement of the colour filters on the pixel array.
    Returns
    -------
    :class:`numpy.ndarray`
        *Bayer* CFA mosaic.
    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array([[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])
    >>> mosaicing_CFA_Bayer(RGB)
    array([[ 0.,  1.],
           [ 1.,  2.]])
    >>> mosaicing_CFA_Bayer(RGB, pattern="BGGR")
    array([[ 2.,  1.],
           [ 1.,  0.]])
    """

    RGB = as_float_array(RGB)

    R, G, B = tsplit(RGB)
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2], pattern)

    CFA = R * R_m + G * G_m + B * B_m

    return CFA