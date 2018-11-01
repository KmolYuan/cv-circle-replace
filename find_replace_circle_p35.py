# -*- coding: utf-8 -*-

from typing import Tuple, Iterator
import cv2
import numpy as np


def find_circle(img: np.ndarray) -> Iterator[Tuple[Tuple[float, float], float]]:
    """Find circle from gray image."""
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, minRadius=80)
    for x, y, r in np.uint16(np.around(circles[0, :, :]))[:]:
        yield (x, y), r


def cropped(img: np.ndarray, x: float, y: float, r: float) -> np.ndarray:
    """Cropped image."""
    return img[(y - r):(y + r), (x - r):(x + r)]


def scaled(img: np.ndarray, factor: float) -> np.ndarray:
    """Scaled image."""
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)


def replace_circle(src_path: str, dist_path: str, show_cropped: bool):
    """Replace src image to dist image."""
    # Source image.
    src_img = cv2.imread(src_path)
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # Find a circle.
    (o_x, o_y), o_r = next(find_circle(src_gray))

    # Create crop mask.
    mask = np.zeros(src_img.shape, dtype=src_img.dtype)
    cv2.circle(mask, (o_x, o_y), o_r, (255, 255, 255), -1)
    ball_mask = cropped(-(src_img * mask - mask + 255), o_x, o_y, o_r)
    crop_mask = cropped(mask, o_x, o_y, o_r)

    # Draw the circle.
    cv2.circle(src_img, (o_x, o_y), o_r, (0, 0, 255))

    # Destination image.
    dist_img = cv2.imread(dist_path)
    replaced_img = dist_img.copy()
    dist_gray = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)
    for (x, y), r in find_circle(dist_gray):
        # Apply crop mask.
        f_r = r / o_r
        condition = (scaled(crop_mask, f_r) != [0, 0, 0]).all(axis=2)
        cropped(replaced_img, x, y, r)[condition] = scaled(ball_mask, f_r)[condition]
        # Draw the circle.
        cv2.circle(dist_img, (x, y), r, (0, 0, 255))

    # Show images.
    cv2.imshow("Source: {}".format(src_path), src_img)
    cv2.imshow("Destination: {}".format(dist_path), dist_img)
    if show_cropped:
        cv2.imshow("Cropped image", ball_mask)
    cv2.imshow("Replaced image", replaced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    replace_circle('earth.png', 'billiard.jpg', show_cropped=True)
