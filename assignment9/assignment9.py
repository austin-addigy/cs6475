# ASSIGNMENT 9
# Your Name

import numpy as np
import scipy as sp
import cv2

import random

""" Assignment 9 - Building an HDR Image

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


def linearWeight(pixel_value):
    """ Linear weighting function based on pixel intensity

    The weighting function should form a triangle, with a value of zero
    at the min and max pixel intensity values (0 and 255 for uint8), and
    a single maximum at the middle pixel intensity value (127.5 for uint8
    inputs).  It should return either the current pixel intensity value
    (pixel_value) or pixel_range_max - pixel_value, whichever is smaller.

    See the paper, "Recovering High Dynamic Range Radiance Maps from
    Photographs" by Debevec & Malik (available in the course resources
    material on T-Square)

    Args:
    ----------
        pixel_value : np.uint8
            A value from 0 to 255 representing a pixel intensity

    Returns:
    ----------
        weight : np.float64
            A value from 0.0 to pixel_range_max
    """
    max_intensity = 255.  # maximum intensity value of a uint8 picture
    return np.min([pixel_value, max_intensity - pixel_value])

def sampleIntensities(images):
    """ Randomly sample pixel intensity exposure slices for each possible pixel
    intensity value from the exposure stack.

    The returned intensity_values is an array with one row for every possible 
    pixel value, and one column for each image in the exposure stack. The
    values in the array are filled according to the instructions below.
    The middle image of the exposure stack is used to search for
    candidate locations because it is expected to be the least likely
    image where pixels will be over- or under-exposed.

    For each possible pixel intensity level:

        1. Find the location of all pixels in the middle image that
        have the current target intensity

            a. If there are no pixels with the target intensity,
            do nothing

            b. Otherwise, use a uniform distribution to select a
            location from the candidate pixels then, set the
            current row of intensity_values to the pixel intensity
            of each image at the chosen location.

    NOTE: Recall that array coordinates (row, column) are in the opposite
    order of the Cartesian coordinates (x, y) we are all used to.

    Args:
    ----------
        images : list<numpy.ndarray>
            A list containing a stack of single-channel (i.e., grayscale)
            layers of an HDR exposure stack

    Returns:
    ----------
        intensity_values : numpy.array, dtype=np.uint8
            An array containing a uniformly sampled intensity value from each
            exposure layer (shape = num_intensities x num_images)

    """
    images = np.array(images)
    # There are 256 intensity values to sample for uint8 images in the
    # exposure stack - one for each value [0...255], inclusive
    num_intensities = 256
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    # Find the middle image -- use this as the source for pixel intensity locations
    mid = np.round(num_images // 2)  # using integer division is arbitrary in this case
    mid_img = images[mid]

    for i in range(num_intensities):
        idx = np.where(mid_img == i)
        if idx[0].size == 0:
            continue
        pick = np.random.randint(0, idx[0].size)
        intensity_values[i] = images[:, idx[0][pick], idx[1][pick]]

    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    """ Find the camera response curve for a single color channel

    The response curve is obtained by finding the least-squares solution
    to an overdetermined set of constraint equations (i.e., solve for x
    in the linear algebra system of equations Ax=b).

    The first step is to fill in mat_A and mat_b with the constraints
    described in section 2.1 of the research paper, "Recovering High Dynamic
    Range Radiance Maps from Photographs" by Debevec & Malik (available in
    the course resources material on T-Square).

    We recommend that you break the process up into three steps:
        - Fill in data-fitting constraints
        - Fill in smoothing constraints
        - Add the color curve adjustment constraint

    The steps required are described in detail below.

    PART 1: Constraints

        In this part, you will fill in mat_A and mat_b with coefficients
        corresponding to constraint equations for the response curve.

        WHAT TO DO?
        1a. Data-Fitting Constraints

            Data-fitting constraints correspond to observed intensity values
            in the image stack.  These constraints ensure that the response
            curve correctly predicts the observed data samples from the
            exposure stack.

            The intensity_samples array contains one row for each possible
            pixel intensity, and one row for each image in the exposure
            stack.

            The data fitting constraints fill the first NxP rows of mat_A,
            where N is the number of rows in intensity_samples, and P is
            the number of pictures in the exposure stack.

            For each row i and column j of the intensity_samples array:

                NOTE: The value of idx should start at zero and increment
                      by one each time j is incremented.

                i.   Set the value of mat_A at (idx, intensity_ij) to
                     the value of the weighting function evaluated on
                     the intensity_ij (w_ij)

                ii.  Set the value of mat_A at (idx, num_samples + i)
                     to the negative value of the weighting function
                     evaluated on the intensity_ij (-w_ij)

                iii. Set mat_b at (idx, 0) to the product of the value
                     of the weighting function and the log of exposure j
                     (wij * log_exposure[j])

        1b. Smoothing Constraints

            Smoothing constraints ensure that the response curve is smooth
            by penalizing changes in the second derivative of the function;
            ideally, the second derivative would be zero everywhere, so
            the constraints in this section are all equations that equal
            zero, and there is no need to set any values in mat_b.

            The smoothing constraints are entered starting from an offset of
            NxP + 1 rows in mat_A, i.e., the first row after the last data
            constraint. The number of constraints is two less than the number
            of possible pixel values (e.g., 254 rows for uint8 images
            because they have 256 possible values).

            For each value idx in the closed range (i.e., including both
            endpoints) [1...intensity_range - 1]:

                NOTE: The indices for the smoothing constraints shown here
                      correct for off-by-one since this loop starts at value
                      1 and python is 0-indexed.

                i.   Set mat_A at (offset + idx - 1, idx - 1) to the product
                     of the smoothing lambda parameter and the value of the
                     weighting function for idx.

                     (smoothing_lambda * weighting_function(idx))

                ii.  Set mat_A at (offset + idx - 1, idx) to -2 times the
                     product of the smoothing lambda parameter and the value
                     of the weighting function for idx.

                     (-2 * smoothing_lambda * weighting_function(idx))

                iii. Set mat_A at (offset + idx - 1, idx + 1) to the product
                     of the smoothing lambda parameter and the value of the
                     weighting function for idx.

                     (smoothing_lambda * weighting_function(idx))

        1c. Color curve constraint

            This constraint corresponds to the assumption that middle value
            pixels have unit exposure. This constraint is an equation that
            is equal to zero, so there is no need to set any values in mat_b.

            Set the value of mat_A in the last row and middle column
            (mat_A.shape[0], intensity_range // 2) to the constant 1.

    PART 2: Solving the system

        In this part we do some simple linear algebra to solve the system Ax=b

        Ax = b
        A^-1 * A * x = b    NOTE: The * operator here is the dot product, but
                                  the numpy * operator performs an element-wise
                                  multiplication so don't use it -- use np.dot.
        x = A^-1 * b

        Unfortunately, we can't obtain the matrix inverse is only defined for 
        square matrices, and our system has more constraint equations than
        unknown variables because the system is overdetermined. However, we
        can however use a different method called the Moore-Penrose
        Pseudoinverse to find the least-squares fit.

        WHAT TO DO?
        1a. Get the pseudoinverse of A. Numpy has an implementation of the
            Moore-Penrose Pseudoinverse, so this is just a function call.

        1b. Multiply that psuedoinverse -- dot -- b. This becomes x. Make sure
            x is of the size 512 x 1.

    NOTE: For those of you unfamiliar with Python and getting to learn it
    this semester, this will have something "weird". `weighting_function` is
    not a value, but rather a function. This means we pass in the name of a
    function and then within the computeResponseCurve function you can use it
    to compute the weight (so you can do weighting_function(10) and it will
    return a value for the weight). Feel free to ask questions on Piazza if
    that doesn't make sense.

    Args:
    ----------
        intensity_samples : numpy.ndarray
            Stack of single channel input values (num_samples x num_images)

        log_exposures : numpy.ndarray
            Log exposure times (size == num_images)

        smoothing_lambda : float
            A constant value to correct for scale differences between
            data and smoothing terms in the constraint matrix -- source
            paper suggests a value of 100.

        weighting_function : callable
            Function that computes the weights

    Returns:
    ----------
        g : numpy.ndarray, dtype=np.float64
            g(x) is the log exposure corresponding to pixel intensity value z

    """
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range,
                      num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # PART 1a: Create data-fitting constraints
    idx = 0
    for i, ir in enumerate(intensity_samples):
        for j, jc in enumerate(ir):
            w_ij = weighting_function(jc)
            mat_A[idx, jc] = w_ij
            mat_A[idx, num_samples + i] = -w_ij
            mat_b[idx, 0] = w_ij * log_exposures[j]
            idx = idx + 1

    # PART 1b: Apply smoothing constraints throughout the pixel range. Loop
    # 1 (skip first value) to intensity range (last loop value is
    # intensity_range-1 -- i.e., skip the last value). Remember to
    # offset the smoothing constraint rows past the data constraints.
    # i.e., the first row should be after offset = num_samples * num_images
    offset = num_images * num_samples
    for idx in range(1, intensity_range):
        w = weighting_function(idx)
        mat_A[offset + idx - 1, idx - 1] = smoothing_lambda * w
        mat_A[offset + idx - 1, idx] = -2 * smoothing_lambda * w
        mat_A[offset + idx - 1, idx + 1] = smoothing_lambda * w

    # PART 1c: Adjust color curve by adding a constraint forcing the middle pixel
    # value to be zero
    mat_A[-1, intensity_range // 2] = 1

    # PART 2: Solve the system using x = A^-1 * b
    x = np.linalg.lstsq(mat_A, mat_b)[0]

    # Assuming that you set up your equation so that the first elements of
    # x correspond to g(z); otherwise change to match your constraints
    g = x[0:intensity_range + 1]

    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    """ Use the response curve to calculate the radiance map for each pixel
    in the current color layer.

    Once you have the response curve, you can use it to recover the radiance
    of each pixel in the scene. The process is described below:

    1. Initialize every pixel of the output layer to 1.0.  Note that the
       radiance map is computed in log space, where a value of 1.0 is
       equivalent to a value of 0.0 in intensity space.

    2. For each location i, j in the image:

        a. Get the pixel value from each image in the exposure stack at
           the current location (i, j)

        b. Calculate the weight of pixel[i][j] in each image from the
           exposure stack using weighting_function

        c. Calculate the sum of all weights at (i, j)

        d. If the sum of the weights is > 0.0, set the value of the output
           array at location (i, j) equal to the weighted sum of the
           difference between the response curve evaluated for the pixel
           value from each image in the exposure stack and the log of the
           exposure time for that image, divided by the sum of the weights
           for the current location

           output[i][j] = sum(response_curve[pixel_vals] - log_exposure_times) / sum_weights

    Args:
    ----------
        images : list
            Collection containing a single color layer (i.e., grayscale)
            from each image in the exposure stack. (size == num_images)

        log_exposure_times : numpy.ndarray
            Array containing the log exposure times for each image in the
            exposure stack (size == num_images)

        response_curve : numpy.ndarray
            Least-squares fitted log exposure of each pixel value z

        weighting_function : callable
            Function that computes the weights

    Returns:
    ----------
        img_rad_map : numpy.ndarray, dtype=np.float64
            The image radiance map in log space
    """
    images = np.array(images)

    min_pixel = 0.0
    max_pixel = 255.0
    img_shape = images[0].shape
    img_rad_map = np.ones(img_shape, dtype=np.float64)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            px = np.array([images[k][i, j] for k in range(images.shape[0])])
            wt = map(weighting_function, images[:, i, j])
            wtsum = sum(wt)
            if wtsum > 0:
                img_rad_map[i, j] = sum(wt * (response_curve[px] - log_exposure_times)) / wtsum
    return img_rad_map

def computeHDR(images, log_exposure_times, smoothing_lambda=100.):
    """ This function combines the functions above to produce the HDR
    computation pipeline.

    NOTE: This function is NOT scored as part of this assignment.  You may
          modify it as you see fit.

    The basic overview is to do the following for each channel:

    1. Sample pixel intensities from random locations through the image stack
       to determine the camera response curve

    2. Compute response curves for each color channel

    3. Build image radiance map from response curves

    4. Apply tone mapping to fit the high dynamic range values into a limited
       range for a specific print or display medium (NOTE: we don't do this
       part except to normalize - but you're free to experiment.)

    Args:
    ----------
        images : list<numpy.ndarray>
            A list containing an exposure stack of images

        log_exposure_times : numpy.ndarray
            The log exposure times for each image in the exposure stack

        smoothing_lambda : np.int (Optional)
            A constant value to correct for scale differences between
            data and smoothing terms in the constraint matrix -- source
            paper suggests a value of 100.

    Returns:
    ----------
        hdr_image : numpy.ndarray
            The resulting HDR image
    """
    images = map(np.atleast_3d, images)
    num_channels = images[0].shape[2]

    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    for channel in range(num_channels):

        # Collect the current layer of each input image from
        # the exposure stack
        layer_stack = [img[:, :, channel] for img in images]

        # Sample image intensities
        intensity_samples = sampleIntensities(layer_stack)

        # Compute Response Curve
        response_curve = computeResponseCurve(intensity_samples,
                                              log_exposure_times,
                                              smoothing_lambda,
                                              linearWeight)

        # Build radiance map
        img_rad_map = computeRadianceMap(layer_stack,
                                         log_exposure_times,
                                         response_curve,
                                         linearWeight)

        # We don't do tone mapping, but here is where it would happen. Some
        # methods work on each layer, others work on all the layers at once;
        # feel free to experiment.
        hdr_image[..., channel] = cv2.normalize(img_rad_map, alpha=0, beta=255,
                                                norm_type=cv2.NORM_MINMAX)

    return hdr_image
