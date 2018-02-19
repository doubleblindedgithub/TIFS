import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from docrec.image.processing.morphology import remove_lines

k = 0.0393700787401575
R = 400 * k 

fig = plt.figure(figsize=(12, 12), dpi=300)
ax = fig.add_subplot(111)

# Open test image
assert len(sys.argv) > 1

# Open as graysacle
img = cv2.imread(sys.argv[1], 0)

# Remove noise
img = cv2.medianBlur(img, 5, 5)

# Remove lines
img = remove_lines(img)

ax.imshow(img, cmap=plt.get_cmap('gray'))
ax.axis('off')    
plt.show()