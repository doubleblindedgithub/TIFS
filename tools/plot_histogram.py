import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

assert len(sys.argv) > 1

img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img, 5)
hist = np.bincount(img.flatten(), minlength=256)
hist = np.cumsum(hist[: 250])
plt.plot(hist)
plt.show()
