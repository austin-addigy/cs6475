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

import assignment7 as a7


SRC_FOLDER = "images/source"
OUT_FOLDER = "images/output"


def main(image_files, output_folder):
    """
    Find matches between images and save the annotated matches
    """

    template = cv2.imread(image_files['template'])
    del image_files['template']

    for transform_name in image_files:
        print "    Processing {} image".format(transform_name)

        image = cv2.imread(image_files[transform_name])

        keypoints1, keypoints2, matches = a7.findMatchesBetweenImages(template, image)
        annotated_matches = a7.drawMatches(template, keypoints1, image, keypoints2, matches)
        cv2.imwrite(path.join(output_folder, transform_name + '.jpg'), annotated_matches)  


if __name__ == "__main__":
    """
    Apply pyramid blending to all folders below SRC_FOLDER that contain
    a black, white, and mask image.
    """

    EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
    NAME_KEYS = ['template', 'lighting', 'rotation', 'sample', 'scale']

    subfolders = os.walk(SRC_FOLDER)
    subfolders.next()  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(OUT_FOLDER, image_dir)
        image_files = {}  # map transform name keys to image file names

        print "Processing files in '" + image_dir + "' folder..."

        try:

            for name in NAME_KEYS:
                file_list = reduce(list.__add__, map(glob,
                    [os.path.join(dirpath, '*{}.'.format(name) + ext) for ext in EXTENSIONS]))

                if len(file_list) == 0:
                    msg = "  Unable to proceed: no file named {} found in {}"
                    raise RuntimeError(msg.format(name, dirpath))
                elif len(file_list) > 1:
                    msg = "  Unable to proceed: too many files matched the pattern `*{}.EXT` in {}"
                    raise RuntimeError(msg.format(name, dirpath))

                image_files[name] = file_list[0]

        except RuntimeError as err:
            print err
            continue  # skip this folder

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        main(image_files, output_dir)
