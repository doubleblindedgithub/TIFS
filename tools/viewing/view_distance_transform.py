import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import stats
from docrec.image.processing.threshold import threshold_hasan_karan
from docrec.image.processing.blur import blur_hasan_karan

img_filename = sys.argv[1]

fig = plt.figure(figsize=(12, 12), dpi=300)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Open as graysacle
gray = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
blur = blur_hasan_karan(gray)
_, bw = threshold_hasan_karan(255 - blur, 1)


dist = cv2.distanceTransform(
    bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
)

print dist 

# Open as rgb
rgb = cv2.imread(img_filename, cv2.IMREAD_COLOR)


#
#counts = np.bincount(gray.flatten())
#probs = counts / float(counts.sum())
#
## Stats
#entropy = stats.entropy(probs)
#skew = stats.skew(probs)
#kurtosis = stats.kurtosis(probs)
#weber_contrast = (255 - gray).sum() / (255.0 * gray.size)
#print('Shannon entropy % .5f' % entropy)
#print('Skew % .5f' % skew)
#print('Kurtosis % .5f' % kurtosis)
#print('Weber contrast % .5f' % weber_contrast)
ax1.imshow(dist)
ax1.axis('off')
ax2.imshow(rgb)
ax2.axis('off')
plt.show()