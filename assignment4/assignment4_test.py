import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import assignment4 as a4

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "test_images"


class Assignment4Test(unittest.TestCase):

    def setUp(self):
        self.testImage = cv2.imread(path.join(IMG_FOLDER, "butterfly.jpg"),
                                    cv2.IMREAD_GRAYSCALE)

        if self.testImage is None:
            raise IOError("Error, image test_image.jpg not found.")


if __name__ == '__main__':
    unittest.main()