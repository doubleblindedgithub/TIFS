import sys
sys.path.append('../')

import os
import numpy as np
import cv2
import cPickle as pickle
from docrec.strips.strips import Strips
import matplotlib.pyplot as plt
assert len(sys.argv) > 1

margins = pickle.load(open('../margins_marques_2013.pkl', 'r'))
dataset_path = sys.argv[1]
docs = os.listdir(dataset_path)
blank = []
blank_thresh = []
noblank = []
noblank_thresh = []
L = 200
for doc in docs:
    path = os.path.join(dataset_path, doc)
    strips = Strips(path)
    n = len(strips.strips)
    left = margins[doc]['left']
    right = margins[doc]['right']
#    left = 0
#    right = 0
    for i, strip in enumerate(strips.strips[left : n - right], left + 1):
        image = strip.filled_image()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        noblank.append((image <= L).sum())
        val, _= cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if val == 0:
            print i, doc
        noblank_thresh.append(val)
        
    for strip in strips.strips[: left] + strips.strips[n - right : ]:
        image = strip.filled_image()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        blank.append((image <= L).sum())
        val, _= cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print val
        blank_thresh.append(val)

hist_blank = np.bincount(blank, minlength=max(blank))
hist_noblank = np.bincount(noblank, minlength=max(noblank))
hist_blank_thresh = np.bincount(blank_thresh, minlength=256)
hist_noblank_thresh = np.bincount(noblank_thresh, minlength=256)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(hist_blank)
ax2.plot(hist_noblank)
ax3.plot(hist_blank_thresh)
ax4.plot(hist_noblank_thresh)
plt.show()
