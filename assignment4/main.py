
import cv2
import numpy as np
import scipy as sp

from os import path

import assignment4 as a4

"""
You can use this file to write code for part 2. You are NOT required
to use this file, and ARE ALLOWED to make ANY changes you want in
THIS file. This file will not be submitted with your assignment
or report, so if you write code for above & beyond effort, make sure
that you include important snippets in your writeup. CODE ALONE IS
NOT SUFFICIENT FOR ABOVE AND BEYOND CREDIT.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "test_images"

GAUSSIAN_KERNEL = np.array([[  1,  4,  6,  4,  1],
                            [  4, 16, 24, 16,  4],
                            [  6, 24, 36, 24,  6],
                            [  4, 16, 24, 16,  4],
                            [  1,  4,  6,  4,  1]])


def find_edges(image):
    # TODO: implement your own edge detector
    pass


def main():
    # TODO: replace this code with your own

    img = cv2.imread(path.join(IMG_FOLDER, "butterfly.jpg"), cv2.IMREAD_GRAYSCALE)
    edge_img = find_edges(img)
    norm_img = a4.normalizeImage(edge_img)
    cv2.imwrite("null_result.png", norm_img)


if __name__ == "__main__":
    main()