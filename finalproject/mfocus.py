import cv2
import numpy as np

# [1]
def norm_brightness(imstacks):
	imstack, imstackBW = imstacks

	out = [imstack[0]]
	m1 = np.mean(imstackBW[0])

	for i, img in enumerate(imstack[1:]):
		m2 = np.mean(imstackBW[i+1])
		out.append(img + (m1 - m2))

	return [out, imstackBW]

def find_homography(kp1, kp2, matches):
    pt1 = np.array([[kp1[match.queryIdx].pt] for match in matches], dtype=np.float_)
    pt2 = np.array([[kp2[match.trainIdx].pt] for match in matches], dtype=np.float_)
    
    H, _ = cv2.findHomography(pt1, pt2, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    
    return H

def align_images(stacks):
    imstack, imstackBW = stacks

    out = [imstack[0]]

    orb = cv2.ORB()
    kp1, des1 = orb.detectAndCompute(imstackBW[0], None)    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i, img in enumerate(imstack[1:]):
        kp2, des2 = orb.detectAndCompute(imstackBW[i], None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        H = find_homography(kp1, kp2, matches[:30])
        wimg = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)       ## Do only linear interpolation
        out.append(wimg)

    return [out, imstackBW]

def genKernel(parameter):
    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter, 0.25, 0.25 - parameter /2.0])
    return np.outer(kernel, kernel)

def reduce_layer(image):
    kernel = genKernel(0.4)
    return cv2.filter2D(image, -1, kernel)[::2, ::2]

def expand_layer(image):
    kernel = genKernel(0.4)
    out = np.zeros((image.shape[0]*2, image.shape[1]*2, 3))
    out[::2, ::2, :] = image
    return cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_CONSTANT) * 4

def gaussian_pyramid(image, levels):
    gpyr = [image]
    for i in range(levels):
        bimg = cv2.GaussianBlur(gpyr[-1], (5, 5), 0.05)
        gpyr.append(reduce_layer(bimg))
    return gpyr

def crop(image, size):
    imh, imw = image.shape[:2]
    if not imh == size[0]:
        image = image[:size[0]-imh, :]
    if not imw == size[1]:
        image = image[:, :size[1]-imw]
    return image

def laplacian_pyramid(gpyr):
    gpyr = gpyr[::-1]
    out = [gpyr[0]]
    
    for i, img in enumerate(gpyr[:-1]):
        img = expand_layer(img)
        img = crop(img, gpyr[i+1].shape)
        out.append(gpyr[i+1] - img)

    return out[::-1]

def collapse_pyramid(lpyr):
    s = lpyr[-1]
    for i in range(len(lpyr)-1, 0, -1):
        s = expand_layer(s)
        s = crop(s, lpyr[i-1].shape)
        s = s + lpyr[i-1]
    return s
    
def make_single_image(stack):
    out = stack[0]
    (h, w) = out.shape[:2]
    for img in stack[1:]:
        (nh, nw) = img.shape[:2]
        img = np.concatenate((img, np.zeros((h-nh, nw, 3))), axis=0)
        out = np.concatenate((out, img), axis=1)
    return out

def laplacian(img, ksizeL=5, ksizeM=5, ksizeG=5, sigma=10):
	img = cv2.GaussianBlur(img, (ksizeG, ksizeG), sigma)
	img = cv2.Laplacian(img, cv2.CV_32F, ksize=ksizeL)
	return cv2.medianBlur(img, ksizeM)

def mfocus_laplacian_of_gaussian(imstacks):
	imstack, imstackBW = imstacks

	# laplsum = np.zeros(imstackBW[0].shape, dtype=np.int64)
	laplstack = np.zeros((len(imstackBW),) + imstackBW[0].shape, dtype=np.float32)
	for i, img in enumerate(imstackBW):
		img = laplacian(cv2.dilate(img, np.ones((5, 5))))
		# Avg filter here?
		# laplsum = laplsum + img
		laplstack[i] = img

	out = np.zeros(imstack[0].shape, dtype=np.float32)
	idm = np.zeros(imstackBW[0].shape, dtype=np.uint8)
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			p = np.abs(laplstack[:, i, j])
			idx = np.where(p == max(p))[0][0]
			# out[i, j] = imstack[idx][i, j]
			idm[i, j] = idx

	# st = zip(*np.where(laplsum == max(laplsum)))[0]
	idm = np.around(cv2.GaussianBlur(idm, (5, 5), 10)).astype(np.uint8)

	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			out[i, j] = imstack[idm[i, j]][i, j]

	# out = cv2.medianBlur(out, 5)

	return out

def mfocus_multi_scale_lapl(lpyr_stack):
    bpx = 2
    wsize = bpx * 2 + 1

    lstack = []
    nlevels = len(lpyr_stack[0])
    for i in range(nlevels):
        level = []
        for lpyr in lpyr_stack:
            lpyr = cv2.copyMakeBorder(lpyr[nlevels-i-1], bpx, bpx, bpx, bpx, cv2.BORDER_REFLECT_101)
            level.append(lpyr)
        lstack.append(level)

    new_lpyr = []

    D = -np.inf
    idx = 0
    pyr = np.zeros(lstack[0][0].shape)
    for h in range(len(lstack)):
        for i in range(pyr.shape[h]-2*bpx):
            for j in range(pyr.shape[1]-2*bpx):
                for k, l in enumerate(lstack[h]):
                    patch = l[i:i+wsize, j:j+wsize].astype(np.float32)
                    mean = np.mean(patch)
                    patch = (patch - mean) ** 2 / 25.0
                    d = np.sum(patch)
                    if d > D:
                        D = d
                        idx = k
                pyr[i, j] = lstack[0][idx][i, j]
        new_lpyr.append(pyr)
    
    return new_lpyr

inputs = {
	'flower': (100, 8),
	'piano': (100, 8),
	'objects': (1510, 8),
	'yard': (1519, 8),
	'rocks': (1527, 9),
}

def main():

	algo = mfocus_laplacian_of_gaussian

	file_fmt = '%s/%s/IMG_%d.JPG'

	dir = 'objects'
	start_idx, nphotos = inputs[dir]

	print ('- Reading in images')
	imstacks = [[], []]
	for i in range(start_idx, start_idx + nphotos):
		img = cv2.imread(file_fmt%('input', dir, i))
		imstacks[0].append(img.astype(np.float32))
		imstacks[1].append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

	print('- Normalizing brightness')
	imstacks = norm_brightness(imstacks)

	# print('- Aligning images')
	# imstacks = align_images(imstacks)

	print('- Calling algorithm %s'%algo.__name__)
	out = algo(imstacks)
	cv2.imwrite('out.png', out)

	print('- Writing image(s)')
	out = imstacks[0]
	for i in range(nphotos):
		cv2.imwrite(file_fmt%('output', dir, i), out[i])

if __name__ == '__main__':
	main()