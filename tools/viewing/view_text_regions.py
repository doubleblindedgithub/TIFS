import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import cv2
import numpy as np
from docrec.ocr.text.extraction import *
from scipy import stats


img_filename = sys.argv[1]

dpi = 300
if len(sys.argv) > 2:
    dpi = int(sys.argv[2])

fig = plt.figure(figsize=(12, 12), dpi=dpi)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Open as graysacle
gray = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

# Open as rgb
rgb = cv2.imread(img_filename, cv2.IMREAD_COLOR)

# Extract regions
boxes, labels = extract_text_regions(gray, 30, dpi)
print labels
#boxes, labels = extract_text_regions_hasan_karan(gray, 30, dpi)
#boxes, _ = extract_text_regions_tesseract(gray)

# Result
for x, y, w, h in boxes:
    cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
ax1.imshow(rgb)
ax1.axis('off')    

if labels is not None:    
    ax2.imshow(labels, plt.cm.jet)
ax2.axis('off')

plt.show()