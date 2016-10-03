import cv2
import numpy as np

import assignment4 as a4

GAUSSIAN_KERNEL = np.array([[  1,  4,  6,  4,  1],
                            [  4, 16, 24, 16,  4],
                            [  6, 24, 36, 24,  6],
                            [  4, 16, 24, 16,  4],
                            [  1,  4,  6,  4,  1]])


def find_edges(image):
    
    gx = a4.gradientX(image).astype(np.int64)
    gx = np.hstack((gx, np.zeros((image.shape[0], 1))))
    gy = a4.gradientY(image).astype(np.int64)
    gy = np.vstack((gy, np.zeros((1, image.shape[1]))))
    z = np.sqrt(gx**2 + gy**2)
    z = cv2.GaussianBlur(z, (3, 3), 0.5)
    
    th = 30
    z[z >= th] = 255
    z[z < th] = 0
    
    return z

def main():
    img = cv2.imread("sauce.jpg", cv2.IMREAD_GRAYSCALE)
    edge_img = find_edges(img)
    # edge_img = cv2.Canny(img, 100, 500)
    norm_img = a4.normalizeImage(edge_img)
    cv2.imwrite("edges.png", norm_img)
    cv2.imwrite("orig.png", img)

if __name__ == "__main__":
    main()