import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import stats


img_filename = sys.argv[1]

fig = plt.figure(figsize=(12, 12), dpi=300)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Open as graysacle
gray = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

# Open as rgb
rgb = cv2.imread(img_filename, cv2.IMREAD_COLOR)

counts = np.bincount(gray.flatten())
probs = counts / float(counts.sum())

# Stats
entropy = stats.entropy(probs)
skew = stats.skew(probs)
kurtosis = stats.kurtosis(probs)
weber_contrast = (255 - gray).sum() / (255.0 * gray.size)
print('Shannon entropy % .5f' % entropy)
print('Skew % .5f' % skew)
print('Kurtosis % .5f' % kurtosis)
print('Weber contrast % .5f' % weber_contrast)
ax1.plot(probs)
ax2.plot(probs.cumsum())
plt.show()