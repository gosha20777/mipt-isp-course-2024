from scipy.ndimage.filters import convolve, convolve1d

from colour.hints import ArrayLike, Literal, NDArray
from colour.utilities import as_float_array, tstack

from .utils import masks_CFA_Bayer

import numpy as np

def _cnv_h(x: ArrayLike, y: ArrayLike) -> NDArray:
    """Perform horizontal convolution."""

    return convolve1d(x, y, mode="mirror")


def _cnv_v(x: ArrayLike, y: ArrayLike) -> NDArray:
    """Perform vertical convolution."""

    return convolve1d(x, y, mode="mirror", axis=0)


def bayer2rgb(
    CFA: ArrayLike,
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"],
) -> NDArray:
    """
    Return the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    DDFAPD - *Menon (2007)* demosaicing algorithm.
    Parameters
    ----------
    CFA
        *Bayer* CFA.
    pattern
        Arrangement of the colour filters on the pixel array.
    refining_step
        Perform refining step.
    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace array.
    References
    ----------
    :cite:`Menon - Demosaicing With Directional Filtering and a
    posteriori Decision`
    Examples
    --------
    >>> import numpy as np
    >>> CFA = np.array(
    ...     [
    ...         [0.3764706, 0.36078432, 0.40784314, 0.3764706],
    ...         [0.35686275, 0.30980393, 0.36078432, 0.29803923],
    ...     ]
    ... )
    >>> demosaicing_CFA_Bayer_Menon2007(CFA, "BGGR")
    array([[[ 0.30588236,  0.35686275,  0.3764706 ],
            [ 0.30980393,  0.36078432,  0.39411766],
            [ 0.29607844,  0.36078432,  0.40784314],
            [ 0.29803923,  0.3764706 ,  0.42352942]],
           [[ 0.30588236,  0.35686275,  0.3764706 ],
            [ 0.30980393,  0.36078432,  0.39411766],
            [ 0.29607844,  0.36078432,  0.40784314],
            [ 0.29803923,  0.3764706 ,  0.42352942]]])
    """

    CFA = as_float_array(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = as_float_array([0.0, 0.5, 0.0, 0.5, 0.0])
    h_1 = as_float_array([-0.25, 0.0, 0.5, 0.0, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = np.where(G_m == 0, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G_m == 0, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)

    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)

    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(
        C_V
        - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[  # pyright: ignore
            2:, :
        ]
    )

    del h_0, h_1, CFA, C_V, C_H

    k = as_float_array(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )

    d_H = convolve(D_H, k, mode="constant")
    d_V = convolve(D_V, np.transpose(k), mode="constant")

    del D_H, D_V

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)

    del d_H, d_V, G_H, G_V

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)

    k_b = as_float_array([0.5, 0, 0.5])

    R = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
        R,
    )

    R = np.where(
        np.logical_and(G_m == 1, B_r == 1) == 1,
        G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
        R,
    )

    B = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
        B,
    )

    B = np.where(
        np.logical_and(G_m == 1, R_r == 1) == 1,
        G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
        B,
    )

    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(
            M == 1,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )

    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(
            M == 1,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    RGB = tstack([R, G, B])

    del R, G, B, k_b, R_r, B_r

    del M, R_m, G_m, B_m

    return RGB