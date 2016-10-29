"""
You can use this file to execute your code. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code for above & beyond effort, make sure
that you include important snippets in your writeup. CODE ALONE IS
NOT SUFFICIENT FOR ABOVE AND BEYOND CREDIT.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""
import cv2
import numpy as np

import os
import errno

from os import path

import assignment9 as a9


# Change the source folder and exposure times to match your own
# input images. Note that the response curve is calculated from
# a random sampling of the pixels in the image, so there may be
# variation in the output even for the example exposure stack
SRC_FOLDER = "input/sample"
EXPOSURE_TIMES = np.float64([1 / 160.0, 1 / 125.0, 1 / 80.0,
                             1 / 60.0, 1 / 40.0, 1 / 15.0])

OUT_FOLDER = "output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def main(image_files, output_folder, exposure_times, resize=False):
    """ Generate an HDR from the images in the source folder """

    image_files = sorted(image_files)

    # Print the information associated with each image -- use this
    # to verify that the correct exposure time is associated with each
    # image, or else you will get very poor results
    print "{:^30} {:>15}".format("Filename", "Exposure Time")
    print "\n".join(["{:>30} {:^15.4f}".format(*v) for v in zip(image_files, exposure_times)])

    img_stack = [cv2.imread(name) for name in image_files
                 if path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    if not all(img_stack):
        raise RuntimeError("One or more input files failed to load.")

    # Subsampling the images can reduce runtime for large files
    if resize:
        img_stack = [img[::4, ::4] for img in img_stack]

    log_exposure_times = np.log(exposure_times)
    hdr_image = a9.computeHDR(img_stack, log_exposure_times)
    cv2.imwrite(path.join(output_folder, "output.png"), hdr_image)

    print "Done!"


if __name__ == "__main__":
    """
    Generate a panorama from the images in the SRC_FOLDER directory
    """
    np.random.seed()  # set a fixed seed if you want repeatable results

    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = src_contents.next()

    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    print "Processing '" + image_dir + "' folder..."

    image_files = [os.path.join(dirpath, name) for name in fnames]

    main(image_files, output_dir, EXPOSURE_TIMES, resize=False)
