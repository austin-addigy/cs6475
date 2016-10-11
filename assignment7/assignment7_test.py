import cv2
import numpy as np
import scipy as sp
import unittest

from os import path

import assignment7 as a7

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.

    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""

IMG_FOLDER = "images/source/sample"


class Assignment6Test(unittest.TestCase):

    def setUp(self):
        images = {}
        images['template'] = cv2.imread(path.join(IMG_FOLDER, "template.jpg"))
        images['lighting'] = cv2.imread(path.join(IMG_FOLDER, "lighting.jpg"))
        images['rotation'] = cv2.imread(path.join(IMG_FOLDER, "rotation.jpg"))
        images['sample'] = cv2.imread(path.join(IMG_FOLDER, "sample.jpg"))
        images['scale'] = cv2.imread(path.join(IMG_FOLDER, "scale.jpg"))

        if not all(images.values()):
            raise IOError("Error, one or more sample images not found.")

        self.images = images


if __name__ == '__main__':
    unittest.main()
