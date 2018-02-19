import sys
sys.path.append('../../')

import cv2
import numpy as np
from docrec.image.processing.threshold import threshold_hasan_karan
from skimage.filters import rank
from skimage import morphology

img_filename = sys.argv[1]

# Open as graysacle
gray = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
_, thresh = threshold_hasan_karan(255 - gray, 1)
cv2.imwrite('thresh.tif', thresh)