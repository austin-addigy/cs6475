import errno
import os
import sys

import numpy as np
import cv2

from glob import glob

import assignment11 as a11


def vizDifference(diff):
    """ This function normalizes the difference matrices so that they can be
    shown as images.
    """
    return (((diff - diff.min()) /
             (diff.max() - diff.min())) * 255).astype(np.uint8)


def runTexture(img_list, alpha):
    """ This function administrates the extraction of a video texture from the
    given frames, and generates the three viewable difference matrices.
    """
    video_volume = a11.videoVolume(img_list)
    ssd_diff = a11.computeSimilarityMetric(video_volume)
    transition_diff = a11.transitionDifference(ssd_diff)

    print "Alpha is {}".format(alpha)
    idxs = a11.findBiggestLoop(transition_diff, alpha)

    diff3 = np.zeros(transition_diff.shape, float)

    for i in range(transition_diff.shape[0]):
        for j in range(transition_diff.shape[1]):
            diff3[i, j] = alpha * (i - j) - transition_diff[i, j]

    return (vizDifference(ssd_diff),
            vizDifference(transition_diff),
            vizDifference(diff3),
            a11.synthesizeLoop(video_volume, idxs[0] + 2, idxs[1] + 2))


def readImages(image_dir):
    """ This function reads in input images from a image directory

    Note: This is implemented for you since its not really relevant to
    computational photography (+ time constraints).

    Args:
    ----------
        image_dir : str
            The image directory to get images from.

    Returns:
    ----------
        images : list
            List of images in image_dir. Each image in the list is of type
            numpy.ndarray.

    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(reduce(list.__add__, map(glob, search_paths)))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
              for f in image_files]

    return images


# The following section will run this file, save the three difference matrices
# as images, and complete the video frame extraction into the output folder.
# You will need to modify the alpha value in order to achieve good results.
if __name__ == "__main__":

    # Change alpha here or from the command line for testing
    alpha = 0.25 if len(sys.argv) < 2 else float(sys.argv[-1])
    video_dir = "candle"
    image_dir = os.path.join("videos", "source", video_dir)
    out_dir = os.path.join("videos", "out")

    print "Reading images."
    images = readImages(image_dir)

    print "Computing video texture..."
    diff1, diff2, diff3, out_list = runTexture(images, alpha)

    try:
        os.makedirs(os.path.join(out_dir, video_dir))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    cv2.imwrite(os.path.join(out_dir, '{}diff1.png'.format(video_dir)), diff1)
    cv2.imwrite(os.path.join(out_dir, '{}diff2.png'.format(video_dir)), diff2)
    cv2.imwrite(os.path.join(out_dir, '{}diff3.png'.format(video_dir)), diff3)

    for idx, image in enumerate(out_list):
        cv2.imwrite(os.path.join(out_dir, video_dir,
                    'frame{0:04d}.png'.format(idx)), image)
