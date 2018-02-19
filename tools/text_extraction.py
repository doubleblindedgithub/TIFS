import sys
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt

# Graphics removal
dpi = 300
R = dpi / 25.44
    
# Extent range
min_ext = 1 / 4.5
max_ext = 0.95

# Minimum aspect ratio
min_ar = 0.2

# Text box height range
min_height = 1.2 * R
max_height = 5.5 * R

# Maximum allowed separation for characters
max_sep = 1.2 * R

# Maximum thickness for characters
max_thickness = 0.8 * R

# Graphics maximum height
max_graphics_height = 1.5 * max_height

# Maximum character width variation
max_var = 0.5

fig = plt.figure(figsize=(8, 8), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

assert len(sys.argv) > 1
filename = sys.argv[1]
image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

mask = 255 * np.ones(image.shape[ : 2], dtype=np.uint8)
if len(sys.argv) > 2:
    filename = sys.argv[2]
    mask = np.load(filename)

image = cv2.bitwise_or(image, cv2.bitwise_not(mask))
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Character-level thresholded image
chars = cv2.bitwise_not(thresh)
labels_chars = measure.label(chars)

# Text candidates
dx = int(2 * max_sep)
dy = dx / 2
dx_dy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dx, dy))
text = cv2.morphologyEx(chars, cv2.MORPH_CLOSE, dx_dy)

# Filter out graphics
labels_text = measure.label(text)
props_text = measure.regionprops(labels_text)
#labels_to_remove = set([])
for region_text in props_text:
    yt_min, xt_min, yt_max, xt_max = region_text.bbox
    wt = xt_max - xt_min
    ht = yt_max - yt_min

    if not (min_height <= ht <= max_height):
        labels_to_remove = set(
            labels_chars[yt_min : yt_max, xt_min : xt_max].flatten()
        )
        for label in labels_to_remove:
            labels_chars[labels_chars == label] = 0
    else:
        # Filtering out non-characters candidates
        props_chars = measure.regionprops(
            labels_chars[yt_min : yt_max, xt_min : xt_max]
        )
        for region_char in props_chars:
            # Geometric constraints
            hc, wc = region_char.image.shape
            ar = float(hc) / wc
            if (min_ext <= region_char.extent <= max_ext) or (ar >= min_ar):
                labels_chars[labels_chars == region_char.label] = \
                    region_text.label
            else:
                labels_chars[labels_chars == region_char.label] = 0
    

patches = []
if labels_chars.max() > 0:
    props = measure.regionprops(labels_chars)
    for region in props:
        y, x, _, _ = region.bbox
        h, w = region.image.shape
        box = (x, y, w, h)
        print box
        image = 255 * (region.image).astype(np.uint8)
        patches.append((box, image))

for box, image in patches:
    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()