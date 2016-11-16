# ASSIGNMENT 11
# Your Name

""" Assignment 11 - Video Textures

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
import numpy as np
import scipy as sp
import cv2
import scipy.signal


def videoVolume(images):
    """ Create a video volume (4-d numpy array) from the image list.

    Args:
    ----------
        images : list
            A list of frames. Each element of the list contains a numpy array
            of a colored image. You may assume that each frame has the same
            shape, (rows, cols, 3).

    Returns:
    ----------
        output : numpy.ndarray, dtype = np.uint8
            A 4D numpy array. This array should have dimensions
            (num_frames, rows, cols, 3).
    """
    return np.array(images)


def computeSimilarityMetric(video_volume):
    """
    You need to compute the differences between each pair of frames in the
    video volume. The goal, of course, is to be able to tell how good
    a jump between any two frames might be so that the code you write later
    on can find the optimal loop. The closer the similarity metric is to zero,
    the more alike the two frames are.

    This is done by computing the square root of the sum of the differences
    between each frame in the video volume.  Then, we divide by the average
    of the differences to be resolution independent. (This will help when you
    work with videos of different resolutions)

    Suggested Instructions:

        1. Create a for loop that goes through the video volume. Create a
           variable called cur_frame.

            A. Create another for loop that goes through the video volume
                again. Create a variable called comparison_frame.

                i. Inside this loop, compute this mathematical statement.
                    rssd = sum ( (cur_frame - comparison_frame) ** 2 ) ** 0.5

                ii. Set output[i, j] = rssd

        2.  Divide all the values in output by the average value.  This has
            two benefits: first, it removes any resolution dependencies: the
            same video at two different resolutions will end up with the same
            values.  Second, it distributes the values over a consistent range
            regardless of the video, so the rest of your code is not so exposed
            to the quirks of any given video.

    Hints:

        Remember the matrix is symmetrical, so when you are computing the
        similarity at i, j, its the same as computing the similarity at j, i so
        you don't have to do the math twice. This speeds up the function by 2.

        Also, the similarity at all i,i is always zero, no need to calculate it

    Args:
    ----------
        video_volume : numpy.ndarray
            A 4D numpy array with dimensions (num_frames, rows, cols, 3).

            This can be produced by the videoVolume function.

    Returns:
    ----------
        output : numpy.ndarray, dtype=np.float
            A square 2d numpy array where output[i,j] contains the similarity
            score between all frames [i,j].

            This matrix is symmetrical with a diagonal of zeros.
    """

    output = np.zeros((len(video_volume), len(video_volume)), dtype=np.float)

    for i, frame1 in enumerate(video_volume.astype(np.float_)):
        for j, frame2 in enumerate(video_volume[i+1:]):
            rssd = np.sum((frame1 - frame2) ** 2) ** 0.5
            output[i, i+j+1] = rssd
            output[i+j+1, i] = rssd

    return output / np.average(output)


def transitionDifference(ssd_difference):
    """ Compute the transition costs between frames, taking dynamics into
        account.

    Instructions:
        1. Iterate through the rows and columns of ssd difference, ignoring the
           first two values and the last two values. (Do not pad the matrix.)

            1a. For each value at i, j, multiply the binomial filter of length
                five by the weights starting two frames before until two frames
                after, and take the sum of those products. (You may use the
                binomialFilter5() function provided.)

                i.e. Your weights for frame i are:
                     [weight[i - 2, j - 2],
                      weight[i - 1, j - 1],
                      weight[i, j],
                      weight[i + 1, j + 1],
                      weight[i + 2, j + 2]]

                Multiply that by the binomial filter weights at each i, j to
                get your output.

                It may take a little bit of understanding to get why we are
                computing this, the simple explanation is that to change from
                frame 4 to 5, lets call this ch(4, 5), and we make this weight:

                ch(4, 5) = ch(2, 3) + ch(3, 4) + ch(4, 5) + ch(5, 6) + ch(6, 7)

                This accounts for the weights in previous changes and future
                changes when considering the current frame.

                Of course, we weigh all these sums by the binomial filter, so
                that the weight ch(4, 5) is still the most important one, but
                hopefully that gives you a better understanding.

    Args:
    ----------
        ssd_difference : numpy.ndarray
            A difference matrix as produced by your ssd function.

    Returns:
    ----------
        output : numpy.ndarray
            A difference matrix that takes preceding and following frames into
            account. The output difference matrix should have the same dtype as
            the input, but be 4 rows and columns smaller, corresponding to only
            the frames that have valid dynamics.

    Hint: There is an efficient way to do this with 2d convolution. Think about
          the coordinates you are using as you consider the preceding and
          following frame pairings.
    """

    kernel = np.diag(binomialFilter5())
    output = cv2.filter2D(ssd_difference, -1, kernel)
    return output[2:-2, 2:-2]


def findBiggestLoop(transition_diff, alpha):
    """ Given the difference matrix, find the longest and smoothest loop that
    we can.

    Args:
    ----------
        transition_diff : np.ndarray
            A square 2d numpy array of dtype float. Each cell contains the cost
            of transitioning from frame i to frame j in the input video as
            returned by the transitionDifference function.

        alpha : float
            A parameter for how heavily you should weigh the size of the loop
            relative to the transition cost of the loop. Larger alphas favor
            longer loops, but may have rough transitions. Smaller alphas give
            shorter loops, down to no loop at all in the limit.

    start, end will be the indices in the transition_diff matrix that give the
    maximum score according to the following metric:
        score = alpha * (end - start) - transition_diff[end, start]

    Compute that score for every possible starting and ending index (within the
    size of the transition matrix) and find the largest score.

    Returns:
    ----------
        start : int
            The starting index of the longest loop.

        end : int
            The final index of the longest loop.

    NOTE: Due to the binomial filtering, the start/end index are NOT the same
          as the frame numbers.
    """
    start = 0
    end = 0
    largest_score = 0

    for i in range(len(transition_diff)):
        for j in range(i+1, len(transition_diff[0])):
            score = alpha * (j - i) - transition_diff[i, j]
            if score > largest_score:
                largest_score = score
                start = i
                end = j

    return start, end


def synthesizeLoop(video_volume, start, end):
    """ Pull out the given loop from the input video volume.

    Args:
    ----------
        video_volume : np.ndarray
            A (time, height, width, 3) array, as created by your videoVolume
            function.

        start : int
            The index of the starting frame.

        end : int
            The index of the ending frame.

    Returns:
    ----------
        output : list
            A list of arrays of size (height, width, 3) and dtype np.uint8,
            similar to the original input the videoVolume function.
    """
    return list(video_volume[start:end+1])

def binomialFilter5():
    """ Return a binomial filter of length 5.

    Note: This is included for you to use.

    Returns:
    ----------
        output : numpy.ndarray
            A 5x1 numpy array representing a binomial filter.
    """

    return np.array([1 / 16., 1 / 4., 3 / 8., 1 / 4., 1 / 16.], dtype=float)
