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

import assignment8 as a8


SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def main(image_files, output_folder):
    """ Generate a panorama from the images in the source folder """

    inputs = ((name, cv2.imread(name)) for name in sorted(image_files)
              if path.splitext(name)[-1][1:].lower() in EXTENSIONS)

    # start with the first image in the folder and process each image in order
    name, pano = inputs.next()
    print "\n  Starting with: {}".format(name)
    for name, next_img in inputs:

        if next_img is None:
            print "\nUnable to proceed: {} failed to load.".format(name)
            return

        print "  Adding: {}".format(name)

        kp1, kp2, matches = a8.findMatchesBetweenImages(pano, next_img, 10)
        homography = a8.findHomography(kp1, kp2, matches)
        min_xy, max_xy = a8.getBoundingCorners(pano, next_img, homography)
        pano = a8.warpCanvas(pano, homography, min_xy, max_xy)
        pano = a8.blendImagePair(pano, next_img, -min_xy)

    cv2.imwrite(path.join(output_folder, "output.jpg"), pano)
    print "  Done!"


if __name__ == "__main__":
    """
    Generate a panorama from the images in each subdirectory of SRC_FOLDER
    """

    subfolders = os.walk(SRC_FOLDER)
    subfolders.next()  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        print "Processing '" + image_dir + "' folder..."

        image_files = [os.path.join(dirpath, name) for name in fnames]

        main(image_files, output_dir)
