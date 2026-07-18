import cv2
import numpy as np


def crop_character(image):
    """
    Crop the character tightly and add white padding.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image

    Returns
    -------
    numpy.ndarray
    """

    # Binary inverse threshold
    _, thresh = cv2.threshold(
        image,
        240,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # Find non-zero pixels
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    cropped = image[y:y + h, x:x + w]

    # Add padding
    pad = 20

    cropped = cv2.copyMakeBorder(
        cropped,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return cropped
