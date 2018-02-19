import sys
sys.path.append('../../')

import cv2
import numpy as np
from docrec.image.processing.threshold import threshold_hasan_karan
from skimage.filters import rank
from skimage import morphology

img_filename = sys.argv[1]

# Open as graysacle
rgb = cv2.imread(img_filename, cv2.IMREAD_COLOR)
gray = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
info_map = rank.entropy(gray, morphology.disk(5))
if info_map.max() == info_map.min():
    info_map[:] = 0
else:    
    info_map = 255 * (info_map - info_map.min()) / \
                     (info_map.max() - info_map.min()) # Normalizing
info_map = info_map.astype(np.uint8)
_, thresh = threshold_hasan_karan(info_map, 1)
cv2.imwrite('rgb.tif', rgb)
cv2.imwrite('gray.tif', gray)
cv2.imwrite('rgb.tif', rgb)
cv2.imwrite('entropy.tif', info_map)
cv2.imwrite('thresh.tif', thresh)