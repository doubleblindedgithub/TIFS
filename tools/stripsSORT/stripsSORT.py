import sys
sys.path.append('../../')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from docrec.strips.strips import Strips
from skimage import measure
import matplotlib.patches as patches
import os

fig = plt.figure(figsize=(8, 8), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])

curr_doc = [0]
curr_strip1 = [-1]
strips = [None]
offsets = [None]

# Path to dataset
assert len(sys.argv) > 1

path = sys.argv[1]
docs = os.listdir(path)
ext = os.path.splitext(os.listdir(os.path.join(path, docs[0], 'strips'))[0])[1]

def update_image(d=30):
    
    doc = docs[curr_doc[0]]
    
    print 'Processing %s' % doc
    sys.stdout.flush()
    strips_path = os.path.join(path, doc)
    strips[0] = Strips(strips_path)
    
    
def write():
    
    doc = docs[curr_doc[0]]
    strips_path = os.path.join(path, doc, 'strips')
    masks_path = os.path.join(path, doc, 'masks')
   
        
    for i, strip in enumerate(strips[0].strips):
        print 'Saving strip %d' % i
        image = strip.image
        mask = strip.mask
        basename = '%s%02d%s' % (doc, i + 1, ext)
        filename = os.path.join(strips_path, basename)
        cv2.imwrite(filename, image)
        basename = '%s%02d%s' % (doc, i + 1, '.npy')
        filename = os.path.join(masks_path, basename)
        np.save(filename, mask)


def plot():
    
    ax.clear()
    _, _, offsets[0] = strips[0].plot(ax=ax)
    doc = docs[curr_doc[0]]
    ax.set_title(doc)
    fig.canvas.draw()


def press(event):
    
    if event.key.lower() == 'w': # save
        write()
    elif event.key.lower() == 'right': # next doc
        curr_doc[0] = min(curr_doc[0] + 1, len(docs) - 1)
        update_image()
        plot()
    elif event.key.lower() == 'left': # previous doc
        curr_doc[0] = max(curr_doc[0] - 1, 0)
        update_image()
        plot()


def onclick(event):
    
    if event.inaxes is not None:
        set_strip(int(event.xdata), int(event.ydata))


def set_strip(x, y):
    
    left_coords = \
        [strip.left_borders_coordinates() for strip in strips[0].strips]
    right_coords = \
        [strip.right_borders_coordinates() for strip in strips[0].strips]
    
    for l, r, o in zip(left_coords, right_coords, offsets[0]):
        l += o
        r += o
    
    for j in range(len(strips[0].strips)):
        if y < left_coords[j].size:
            l, r = left_coords[j][y], right_coords[j][y]
            if l <= x <= r:
                if curr_strip1[0] == -1:
                    curr_strip1[0] = j
                    print '1: %d' % j
                else:
                    print '2: %d' % j
                    i = curr_strip1[0]
                    curr_strip1[0] = -1
                    
                    aux = strips[0].strips[i]
                    strips[0].strips[i] = strips[0].strips[j]
                    strips[0].strips[j] = aux
                    plot()
                break

    
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('button_press_event', onclick)

update_image()
plot()
plt.show()
