import cv2
import unittest

import assignment2 as a2

"""
You can use this file as a starting point to write your own unit tests
for this assignment. You are encouraged to discuss testing with your
peers, but you may not share code directly. Your code is scored based
on test cases performed by the autograder upon submission -- these test
cases will not be released.
    DO NOT SHARE CODE (INCLUDING TEST CASES) WITH OTHER STUDENTS.
"""


class Assignment2Test(unittest.TestCase):

    def setUp(self):
        self.testImage = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
        self.testImage2 = cv2.imread("test_image_2.jpg",
                                     cv2.IMREAD_GRAYSCALE)
        if self.testImage is None:
            raise IOError("Error, image test_image.jpg not found.")

        if self.testImage2 is None:
            raise IOError("Error, image test_image_2.jpg not found.")


if __name__ == '__main__':
    unittest.main()