#!/home/thiagopx/anaconda2/bin/python
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

fig = plt.figure(figsize=(12, 12), dpi=300)
ax = fig.add_subplot(111)

# Open test image
assert len(sys.argv) > 1
img = cv2.imread(sys.argv[1], 0)
print img.shape, img.dtype
#print np.unique(img), img.dtype
img = cv2.medianBlur(img, 5)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 11, 2)
print th3
ax.imshow(img, cmap=plt.get_cmap('gray'))

hph = np.logical_not(img).sum(axis=1)
print hph.sum()
lines = hph > 0.15*img.shape[1]
bounded_lines = np.pad(lines, mode='constant', pad_width=1, constant_values=0).astype(np.int8)
diffs = np.diff(bounded_lines)
lines_starts = np.where(diffs == 1)[0]
lines_ends = np.where(diffs == -1)[0] - 1

print lines_starts
print lines_ends

for a, b in zip(lines_starts, lines_ends):
    print a, b

#    for i, (im, box, _, _) in enumerate(boxes):
    ax.add_patch(
        patches.Rectangle(
            (0, a), img.shape[1], b - a + 1,
            facecolor=[1, 0, 0, 0.5], edgecolor='none'
        )
    )

ax.axis('off')    
plt.show()



##!/home/thiagopx/anaconda2/bin/python
#import sys
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from PIL import Image
#from tesserocr import PyTessBaseAPI, RIL
#
#fig = plt.figure(figsize=(12, 12), dpi=300)
#ax = fig.add_subplot(111)
#
## Open test image
#assert len(sys.argv) > 1
#image = Image.open(sys.argv[1])
#with PyTessBaseAPI() as api:
#    api.SetImage(image)
#    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
#    print 'Found {} textline image components.'.format(len(boxes))
#    
#    ax.imshow(image)
#    for i, (im, box, _, _) in enumerate(boxes):
#        ax.add_patch(
#            patches.Rectangle(
#                (box['x'], box['y']), box['w'], box['h'],
#                facecolor=[1, 0, 0, 0.5], edgecolor='none'
#            )
#        )
#
#    ax.axis('off')    
#    plt.show()