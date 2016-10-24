# ASSIGNMENT 8
# Your Name

import numpy as np
import scipy as sp
import scipy.signal
import cv2

from cv2 import ORB

""" Assignment 8 - Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""


def getImageCorners(image):
    """
    Return the x, y coordinates for the four corners of an input image

    NOTE: Review the documentation for cv2.perspectiveTransform (which will be
          used on the output of this function) to see the reason for the
          unintuitive shape of the output array.

    NOTE: When storing your corners, they must be in (X, Y) order
          -- keep this in mind and make SURE you get it right.

    Args:
        image : numpy.ndarray
            Input can be a grayscale or color image

    Returns:
        corners : numpy.ndarray, dtype=np.float32
            Array of shape (4, 1, 2)
    """
    s = image.shape
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    corners[0][0] = (0, 0)
    corners[1][0] = (s[1], 0)
    corners[2][0] = (s[1], s[0])
    corners[3][0] = (0, s[0])
    return corners

def findMatchesBetweenImages(image_1, image_2, num_matches):
    """
    Return the top list of matches between two input images.

    NOTE: You will not be graded for this function. This function is almost
          identical to the function in Assignment 7 (we just parametrized the
          number of matches). We expect you to use the function you wrote in
          A7 here.

    Args:
    ----------
        image_1 : numpy.ndarray
            The first image (can be a grayscale or color image)

        image_2 : numpy.ndarray
            The second image (can be a grayscale or color image)

        num_matches : int
            The number of desired matches. If there are not enough, return
            as many matches as you can.

    Returns:
    ----------
        image_1_kp : list[cv2.KeyPoint]
            A list of keypoint descriptors in the first image

        image_2_kp : list[cv2.KeyPoint]
            A list of keypoint descriptors in the second image

        matches : list[cv2.DMatch]
            A list of matches between the keypoint descriptor lists of
            length no greater than num_matches
    """
    orb = cv2.ORB()
    kp1, des1 = orb.detectAndCompute(image_1, None)
    kp2, des2 = orb.detectAndCompute(image_2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)

    return kp1, kp2, matches[:num_matches]

def findHomography(image_1_kp, image_2_kp, matches):
    """
    Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

    Follow these steps:

        ************************************************************
          Before you start, read the documentation for cv2.DMatch,
          and cv2.findHomography
        ************************************************************

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and a
           mask. Ignore the mask, and return the homography.

    Args:
    ----------
        image_1_kp : list[cv2.KeyPoint]
            A list of keypoint descriptors in the first image

        image_2_kp : list[cv2.KeyPoint]
            A list of keypoint descriptors in the second image

        matches : list[cv2.DMatch]
            A list of matches between the keypoint descriptor lists

    Returns:
    ----------
        homography : numpy.ndarray, dtype=np.float64
            A 3x3 array defining a homography transform between
            image_1 and image_2
    """
    image_1_points = np.array([[image_1_kp[match.queryIdx].pt] for match in matches], dtype=np.float32)
    image_2_points = np.array([[image_2_kp[match.trainIdx].pt] for match in matches], dtype=np.float32)

    H, _ = cv2.findHomography(image_1_points, image_2_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    return H

def getBoundingCorners(image_1, image_2, homography):
    """
    Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use getImageCorners() on image 1 and image 2 to get the corner
           coordinates of each image.

        2. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        3. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

        4. Store the minimum values in min_xy, and the maximum values in max_xy

    Args:
    ----------
        image_1 : numpy.ndarray
            A grayscale or color image

        image_2 : numpy.ndarray
            A grayscale or color image

        homography : numpy.ndarray, dtype=np.float64
            A 3x3 array defining a homography transform between image_1 and
            image_2

    NOTE: The inputs may be either color or grayscale, but they will never be
          mixed; both images will either be color, or both will be grayscale.

    Returns:
    ----------
        min_xy : numpy.ndarray
            2x1 array containing the coordinates of the top left corner of
            the bounding rectangle of a canvas large enough to fit both images

        max_xy : numpy.ndarray
            2x1 array containing the coordinates of the bottom right corner
            of the bounding rectangle of a canvas large enough to fit both
            images
    """
    x_min, x_max = 0, 0
    y_min, y_max = 0, 0
    # WRITE YOUR CODE HERE - YOU ONLY NEED TO DEFINE THE FOUR VALUES:
    # x_min, y_min, x_max, y_max


    # END OF CODING
    min_xy = np.array([x_min, y_min])
    max_xy = np.array([x_max, y_max])
    return min_xy, max_xy
    # END OF FUNCTION


def warpCanvas(image, homography, min_xy, max_xy):
    """
    Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        NOTE: You must explain the reason for multiplying x_min and y_min
              by negative 1 in your writeup.

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Args:
    ----------
        image : numpy.ndarray
            A grayscale or color image

        homography : numpy.ndarray, dtype=np.float64
            A 3x3 array defining a homography transform between two sequential
            images in a panorama sequence

        min_xy : numpy.ndarray
            2x1 array containing the coordinates of the top left corner of a
            canvas large enough to fit the warped input image and the next
            image in a panorama sequence

        max_xy : numpy.ndarray
            2x1 array containing the coordinates of the bottom right corner of
            a canvas large enough to fit the warped input image and the next
            image in a panorama sequence

    Returns:
    ----------
        warped_image : numpy.ndarray
            An array containing the warped input image embedded in a canvas
            large enough to join with the next image in the panorama
    """
    warped_image = None
    size = tuple(np.round(max_xy - min_xy).astype(np.int))  # use in call to cv2.warpPerspective
    # WRITE YOUR CODE HERE


    # END OF CODING
    return warped_image
    # END OF FUNCTION


def blendImagePair(warped_image, image_2, point):
    """
    This function takes in an image that has been warped and an image that
    needs to be incorporated into the warped image at the specified point.

    **************************************************************************
        You MUST replace the basic insertion blend provided here to earn
        credit for this function.  The most common implementation is to
        use alpha blending to take the average between the images for the
       pixels that overlap, but you are encouraged to use other approaches.

        We want you to be creative. You can earn Above & Beyond credit on
         this assignment for particularly spectacular blending functions.
    **************************************************************************

    NOTE: This function is not graded by the autograder. It will be scored
          manually by the TAs.

    Args:
    ----------
        warped_image : numpy.ndarray
            An array containing a warped image (color or grayscale) large
            enough to insert the next image in a panoramic sequence

        image_2 : numpy.ndarray
            A grayscale or color image

        point : numpy.ndarray, dtype=Integer
            The (x, y) coordinates for the top left corner to begin inserting
            image_2 into warped_image

    NOTE: The inputs may be either color or grayscale, but they will never be
          mixed; both images will either be color, or both will be grayscale.

    NOTE: Keep in mind that the blend point is in (X, Y) order.

    Returns:
    ----------
        image : numpy.ndarray
            An array the same size as warped_image containing both input
            images on a single canvas
    """
    output_image = np.copy(warped_image)
    # WRITE YOUR CODE HERE

    # REPLACE THIS WITH YOUR BLENDING CODE.
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2

    return output_image
    # END OF FUNCTION
