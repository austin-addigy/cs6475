import numpy as np
import scipy as sp
import scipy.signal
import cv2

""" Assignment 6 - Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into
    classes, or your own infrastructure. This makes grading very difficult for
    us. Please only write code in the allotted region.
"""


def generatingKernel(parameter):
    """
    Return a 5x5 generating kernel based on an input parameter.

    NOTE: This function is provided for you, do not change it.

    Args:
    ----------
        parameter : float
            The kernel generating parameter in the range [0, 1] used to
            generate a 5-tap filter kernel.

    Returns:
    ----------
        output : numpy.ndarray
            A 5x5 array containing the generated kernel
    """
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter, 0.25, 0.25 - parameter /2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """
    Convolve the input image with a generating kernel of parameter of 0.4 and
    then reduce its width and height by two.

    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.

    For grading purposes, it is important that you use a zero border in the
    convolution and that you include the first row (column), skip the second,
    etc., in the sampling phase.

    You can use any / all library functions, broadcasting rules, and/or
    "advanced" indexing techniques for this function.

    Args:
    ----------
        image : numpy.ndarray
            A grayscale image of shape (r, c). The array may have any data
            type (e.g., np.uint8, np.float64, etc.)

        kernel : numpy.ndarray (Optional)
            A kernel of shape (N, N). The array may have any data
            type (e.g., np.uint8, np.float64, etc.)

    Returns:
    ----------
        output : numpy.ndarray, dtype=np.float64
            An image of shape (ceil(r/2), ceil(c/2)). For instance, if the
            input is 5x7, the output will be 3x4.

    """
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)[::2, ::2].astype(np.float64)

def expand_layer(image, kernel=generatingKernel(0.4)):
    """
    Upsample the image to double the size, and then convolve it with
    a generating kernel of a=0.4.

    Upsampling the image means that every other row and every other column will
    have a value of zero (which is why we apply the convolution after).

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images darken as you apply the convolution.
    You must explain why this happens in your submission PDF.

    Please consult the lectures for a more in-depth discussion of how to
    tackle the expand function.

    You can use any / all library functions, broadcasting rules, and/or
    "advanced" indexing techniques for this function.

    Args:
    ----------
        image : numpy.ndarray
            A grayscale image of shape (r, c). The array may have any data
            type (e.g., np.uint8, np.float64, etc.)

        kernel : numpy.ndarray (Optional)
            A kernel of shape (N, N). The array may have any data
            type (e.g., np.uint8, np.float64, etc.)

    Returns:
    ----------
        output : numpy.ndarray, dtype=np.float64
            An image of shape (2*r, 2*c). For instance, if the input is 3x4,
            then the output will be 6x8.
    """
    out = np.zeros((image.shape[0]*2, image.shape[1]*2))
    out[::2, ::2] = image
    return cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_CONSTANT) * 4

def gaussPyramid(image, levels):
    """
    Construct a pyramid from the image by reducing it by the number of levels
    passed in by the input.

    NOTE: You must use your reduce_layer function to generate the output.

    Args:
    ----------
        image : numpy.ndarray, dtype=float
            An image of dimension (r,c)

        levels : int
            A positive integer that specifies the number of reductions you
            should do. For example, if levels = 0, you should return a list
            containing just the input image. If levels = 1, you should
            perform one reduction. len(output) = levels + 1

    Returns:
    ----------
        output : list
            A list of arrays of dtype np.float. The first element of the
            list (output[0]) is layer 0 of the pyramid (the image itself).
            output[1] is layer 1 of the pyramid (image reduced once), etc.
            We have already included the original image in the output array
            for you. The arrays are of type numpy.ndarray.
    """
    output = [image]

    for i in range(levels):
        bimg = cv2.GaussianBlur(output[-1], (5, 5), 0.05)
        output.append(reduce_layer(bimg))

    return output

def laplPyramid(gaussPyr):
    """
    Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.

    NOTE: You must use expand_layer to generate the output. The Gaussian
    Pyramid that is passed in is the output of your gaussPyramid function.

    NOTE: Sometimes the size of the expanded image will be larger than the
    given layer. You should crop the expanded image to match in shape with
    the given layer. If you do not do this, you will get a 'ValueError:
    operands could not be broadcast together' because you can't subtract
    differently sized matrices.

    For example, if my layer is of size 5x7, reducing and expanding will result
    in an image of size 6x8. In this case, crop the expanded layer to 5x7.

    Args:
    ----------
        gaussPyr : list
            A Gaussian Pyramid as returned by your gaussPyramid function.
            It is a list of numpy.ndarray items.

    Returns:
    ----------
        output : list
            A laplacian pyramid of the same size as gaussPyr. This pyramid
            should be represented in the same way as guassPyr, as a list of
            arrays. Every element of the list now corresponds to a layer of
            the laplacian pyramid, containing the difference between two
            layers of the gaussian pyramid.

            NOTE: The last element of output should be identical to the last
                  layer of the input pyramid since it cannot be subtracted
                  anymore.
    """
    def crop_image(image, size):
        imh, imw = image.shape
        if not imh == size[0]:
            image = image[:size[0]-imh, :]
        if not imw == size[1]:
            image = image[:, :size[1]-imw]
        return image

    gaussPyr = gaussPyr[::-1]
    output = [gaussPyr[0]]

    for i, img in enumerate(gaussPyr[:-1]):
        img = expand_layer(img)
        img = crop_image(img, gaussPyr[i+1].shape)
        output.append(gaussPyr[i+1] - img)

    return output[::-1]

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """
    Blend the two laplacian pyramids by weighting them according to the
    gaussian mask.

    The pyramids will have the same number of levels. Furthermore, each layer
    is guaranteed to have the same shape as previous levels.

    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.

    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.

    NOTE: current_mask, white_image, and black_image are variables that refer
    to the image in the current layer we are looking at. You do this
    computation for every layer of the pyramid.

    Args:
    ----------
        laplPyrWhite : list
            A laplacian pyramid of one image, as constructed by your
            laplPyramid function.

        laplPyrBlack : list
            A laplacian pyramid of another image, as constructed by your
            laplPyramid function.

        gaussPyrMask : list
            A gaussian pyramid of the mask. Each value is in the range [0, 1].

     Returns:
    ----------
        output : list
            A list containing the blended layers of the two laplacian pyramids
    """
    blendedPyr = []
    for i, mask in enumerate(gaussPyrMask):
        out = laplPyrWhite[i] * mask + laplPyrBlack[i] * (1 - mask)
        blendedPyr.append(out)

    return blendedPyr

def collapse(pyramid):
    """
    Collapse an input pyramid.

    Approach this problem as follows, start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.

    NOTE: sometimes expand will return an image that is larger than the next
    layer. In this case, you should crop the expanded image down to the size
    of the next layer. Look into numpy slicing / read our README to do this
    easily.

    For example, expanding a layer of size 3x4 will result in an image of size
    6x8. If the next layer is of size 5x7, crop the expanded image to size 5x7.

    Args:
    ----------
        pyramid : list
            A list of numpy.ndarray images. You can assume the input is taken
            from blend() or laplPyramid().

    Returns:
    ----------
        output : numpy.ndarray, dtype=float
            An image of the same shape as the base layer of the pyramid.
    """
    def crop(image, size):
        imh, imw = image.shape
        if not imh == size[0]:
            image = image[:size[0]-imh, :]
        if not imw == size[1]:
            image = image[:, :size[1]-imw]
        return image

    pyramid = pyramid[::-1]
    out = pyramid[0]

    for i in range(1, len(pyramid)):
        out = crop(expand_layer(out), pyramid[i].shape)
        out = out + pyramid[i]

    return out
