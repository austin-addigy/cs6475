# ASSIGNMENT 4
# Your Name
import cv2
import numpy as np
import scipy as sp

""" Assignment 4 - Detecting Gradients / Edges

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into
    classes, or your own infrastructure. This makes grading very difficult for
    us. Please only write code in the allotted region.
"""


def normalizeImage(src):
    """
    Shift and scale the range of values in src to fit in the interval [0...255]

    This function should shift the range of the input array so that the minimum
    value is equal to 0 and apply a linear scaling to the values in the input
    array such that the maximum value of the input maps to 255.

    The result should be equivalent to the library call:

        cv2.normalize(src, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    (Notice that this means that the output array should have the same value
    type as the input array.)

    NOTE: You MAY NOT use any calls to library functions from openCV, scipy, or
          numpy that perform this function. You MAY use numpy operator
          broadcasting and/or "advanced" indexing techniques.

    Args:
        image (numpy.ndarray): An input array to be normalized.

    Returns:
        output (numpy.ndarray): The input array after shifting and scaling the
                                value range to fit in the interval [0...255]
    """
    img = src.astype(np.float_)
    img = img - img.min()
    img = img * 255.0 / img.max()
    return img.astype(src.dtype)

def gradientX(image):
    """
    Compute the discrete gradient of an image in the X direction

    NOTE: See lectures 02-06 (Differentiating an image in X and Y) for a good
          explanation of how to perform this operation.

    The X direction means that you are subtracting columns:

        F(x, y) = F(x+1, y) - F(x, y)

    NOTE: Array coordinates are given in (row, column) order, which is the
          opposite of the (x, y) convention used for Euclidean coordinates

    NOTE: You MAY NOT use any calls to library functions from openCV, scipy, or
          numpy that perform this function. You MAY use numpy operator
          broadcasting and/or "advanced" indexing techniques.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the X direction. The
                                shape of the output array should have a width
                                that is one column less than the original since
                                no calculation can be done once the last column
                                is reached.
    """
    img = image.astype(np.float_)
    img = img[:,1:] - img[:,:-1]
    return img.astype(np.int16)

def gradientY(image):
    """
    Compute the discrete gradient of an image in the Y direction

    NOTE: See lectures 02-06 (Differentiating an image in X and Y) for a good
          explanation of how to perform this operation.

    The Y direction means that you are subtracting columns:

        F(x, y) = F(x, y+1) - F(x, y)

    NOTE: Array coordinates are given in (row, column) order, which is the
          opposite of the (x, y) convention used for Euclidean coordinates

    NOTE: You MAY NOT use any calls to library functions from openCV, scipy, or
          numpy that perform this function. You MAY use numpy operator
          broadcasting and/or "advanced" indexing techniques.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the Y direction. The
                                shape of the output array should have a height
                                that is one row less than the original since
                                no calculation can be done once the last row
                                is reached.
    """
    img = image.astype(np.float_)
    img = img[1:] - img[:-1]
    return img.astype(np.int16)

def padReflectBorder(image, N):
    """
    This function pads the borders of the input image by reflecting the image
    across the boundaries.

    N is the number of rows or columns that should be added at each border;
    i.e., the output size should have 2N more rows and 2N more columns than
    the input image.

    The values in the input image should be copied to fill the middle of the
    larger array, and the borders should be filled by reflecting the array
    contents as described in the documentation for cv2.copyMakeBorder().

    This function should be equivalent to the library call:

        cv2.copyMakeBorder(image, N, N, N, N, borderType=cv2.BORDER_REFLECT_101)

    Note: BORDER_REFLECT_101 means that the values in the image array are
          reflected across the border. Ex.   gfedcb|abcdefgh|gfedcba

    NOTE: You MAY NOT use any calls to numpy or opencv library functions, but
          you MAY use array broadcasting and "advanced" numpy indexing
          techniques for this function.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.
        N (int): An integer strictly greater than zero representing the number
                 of rows or columns of padding to add at each border.

    Returns:
        output (numpy.ndarray): A copy of the input array with 2N additional
                                rows and columns filled with the values of the
                                input image reflected over the borders.
    """
    img = np.hstack((image[:,1:N+1][:,::-1], image, image[:,-N-1:-1][:,::-1]))
    img = np.vstack((img[1:N+1][::-1], img, img[-N-1:-1][::-1]))
    return img

def crossCorrelation2D(image, kernel):
    """
    This function uses native Python code & loops to compute and return the
    valid region of the cross correlation of an input kernel applied to each
    pixel of the input array.

    NOTE: Lectures 2-05, 2-06, and 2-07 address this concept.

    Recall that for an image F and kernel h, cross correlation is defined as:

        G(i,j) = sum_u=-k..k sum_v=-k..k h[u,v] F[i+u,j+v]

    For k = kernel.shape[0] // 2 function should be equivalent to the call:

        cv2.filter2D(image, cv2.CV_16S, kernel)[k:-k+1, k:-k+1]

    See http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d
    for details.

    Your code must operate on each pixel of the image and kernel individually
    for each step of the computation. (We know this is inefficient, but we want
    to make sure that you understand what is really happening within the more
    efficient library functions that are available.)

    NOTE: You MAY NOT use any numpy, scipy, or opencv library functions,
          broadcasting rules, or "advanced" numpy indexing techniques. You must
          manually loop through the image at each pixel. (Yes, we know this is
          slow and inefficient.)

    NOTE: You MAY assume that kernel will always be a square array with an odd
          number of elements.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.
        kernel (numpy.ndarray): A kernel represented in a numpy array of size
                                (k, k) where k is an odd number strictly
                                greater than zero.

    Returns:
        output (numpy.ndarray): The output image. The size of the output array
                                should be smaller than the original image
                                size by k-1 rows and k-1 columns, where k
                                is the size of the kernel.
    """

    def xcorr(patch, kernel):
        sum = 0
        k = kernel.shape[0]
        for i in range(k):
            for j in range(k):
                sum = sum + patch[i, j] * kernel[i, j]
        return sum

    k = kernel.shape[0]
    nrow, ncol = image.shape                ## 2D, so only single channel
    res = np.zeros((nrow-k+1, ncol-k+1))

    for i in range(nrow-k+1):
        for j in range(ncol-k+1):
            res[i, j] = xcorr(image[i:i+k, j:j+k], kernel)

    return res.astype(np.int16)

def pyFilter2D(image, kernel):
    """
    This function applies the input kernel to the image by performing 2D cross
    correlation on each pixel of the input image.

    NOTE: Lectures 2-05, 2-06, and 2-07 address this concept.

    When padReflectBorder and crossCorrelation are implemented properly, this
    function is equivalent to the library call:

        cv2.filter2D(image, cv2.CV_16S, kernel, achor=(-1,-1), delta=0,
                     borderType=cv2.BORDER_REFLECT_101)

    See http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d
    for details.

    NOTE: This function is not graded in the assignment because it is given to
          you, but you may find it helpful for producing output for your
          report. Separating the functions for padding and cross correlation
          allows the autograder to test them independently.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.
        kernel (numpy.ndarray): A kernel represented in a numpy array of size
                                (k, k) where k is an odd number strictly
                                greater than zero.

    Returns:
        output (numpy.ndarray): An image computed by padding the input image
                                border and then performing cross correlation
                                with the input kernel.
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    padded_image = padReflectBorder(image, kernel.shape[0] // 2)
    filtered_image = crossCorrelation2D(padded_image, kernel)
    return filtered_image
