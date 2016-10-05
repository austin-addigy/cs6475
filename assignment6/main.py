"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code for above & beyond effort, make sure
that you include important snippets in your writeup. CODE ALONE IS
NOT SUFFICIENT FOR ABOVE AND BEYOND CREDIT.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import math
import cv2
import numpy as np

import os
import errno

from os import path
from glob import glob

import assignment6 as a6


SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"


def viz_pyramid(pyramid):
    """
    This function creates a single image out of the given pyramid by making every
    level the same size and stacking the image for each layer vertically.
    """
    shape = pyramid[0].shape[:-1]
    img_stack = [cv2.resize(layer, shape[::-1], interpolation=3) for layer in pyramid]
    return np.vstack(img_stack).astype(np.uint8)


def run_blend(black_image, white_image, mask):
    """
    This function administrates the blending of the two images according to mask.

    Assume all images are float dtype, and return a float dtype.
    """

    # Automatically figure out the size
    min_size = min(black_image.shape)
    depth = int(math.floor(math.log(min_size, 2))) - 4  # at least 16x16 at the highest level

    gauss_pyr_mask = a6.gaussPyramid(mask, depth)
    gauss_pyr_black = a6.gaussPyramid(black_image, depth)
    gauss_pyr_white = a6.gaussPyramid(white_image, depth)

    lapl_pyr_black = a6.laplPyramid(gauss_pyr_black)
    lapl_pyr_white = a6.laplPyramid(gauss_pyr_white)

    outpyr = a6.blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    img = a6.collapse(outpyr)

    return (gauss_pyr_black, gauss_pyr_white, gauss_pyr_mask,
            lapl_pyr_black, lapl_pyr_white, outpyr, [img])


def main(black_image, white_image, mask, out_path):
    """
    Apply pyramid blending to each color channel of the input images
    """

    # Convert to double and normalize the images to the range [0..1]
    # to avoid arithmetic overflow issues
    b_img = np.atleast_3d(black_image).astype(np.float) / 255.
    w_img = np.atleast_3d(white_image).astype(np.float) / 255.
    m_img = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = b_img.shape[-1]

    imgs = []

    for channel in range(num_channels):
        imgs.append(run_blend(b_img[:, :, channel],
                              w_img[:, :, channel],
                              m_img[:, :, channel]))

    try:
        os.makedirs(out_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    names = ['gauss_pyr_black', 'gauss_pyr_white', 'gauss_pyr_mask',
             'lapl_pyr_black', 'lapl_pyr_white', 'outpyr', 'outimg']

    for name, img_stack in zip(names, zip(*imgs)):
        imgs = map(np.dstack, zip(*img_stack))
        stack = [cv2.normalize(im, 0, 255, norm_type=cv2.NORM_MINMAX) for im in imgs]
        cv2.imwrite(path.join(out_path, name + '.png'), viz_pyramid(stack))


if __name__ == "__main__":
    """
    Apply pyramid blending to all folders below SRC_FOLDER that contain
    a black, white, and mask image.
    """

    EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])

    for dirpath, dirnames, fnames in os.walk(SRC_FOLDER):

        if len(fnames) == 0:
            continue

        image_dir = os.path.split(dirpath)[-1]

        black_img_names = reduce(list.__add__, map(glob,
            [os.path.join(dirpath, '*black.' + ext) for ext in EXTENSIONS]))
        white_img_names = reduce(list.__add__, map(glob,
            [os.path.join(dirpath, '*white.' + ext) for ext in EXTENSIONS]))
        mask_img_names = reduce(list.__add__, map(glob,
            [os.path.join(dirpath, '*mask.' + ext) for ext in EXTENSIONS]))

        if any(map(lambda n: len(n) != 1, [black_img_names, white_img_names, mask_img_names])):
            raise RuntimeError("There can only be one black, white, and mask image in each input folder.")

        black_img = cv2.imread(black_img_names[0], cv2.IMREAD_COLOR)
        white_img = cv2.imread(white_img_names[0], cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_img_names[0], cv2.IMREAD_COLOR)

        main(black_img, white_img, mask_img, os.path.join(OUT_FOLDER, image_dir))
