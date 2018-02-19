import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import cv2
from skimage import measure
import numpy as np
from docrec.ocr.text.features import chen_stroke_width
from docrec.image.processing.threshold import threshold_hasan_karan
from docrec.image.processing.label import forward_backward_propagation
from docrec.image.processing.blur import blur_hasan_karan
from docrec.image.metrics.complexity import median_complexity, binary_index

class fiter:
    def __init__(self, seq):
        self.curr = 0
        self.seq = seq

    
    def previous(self):
        return self.go(self.curr - 1)

    def next(self):
        return self.go(self.curr + 1)
    
    def go(self, pos):
        if 0 <= pos < len(self.seq):
            self.curr = pos
        else:
            pos = self.curr
        return self.seq[pos]
    

def plot(region):
    sys.stdout.flush()
    label = region.label
    y1, x1, y2, x2 = region.bbox
    width = x2 - x1
    height = y2 - y1
    
    cropped = img[y1:y2, x1:x2]
    _, cropped_bw = threshold_hasan_karan(255 - cropped, 1)
    if cropped_bw is None:
        print 'Empty'
        return
    
    cropped_blur = blur_hasan_karan(cropped)
    _, cropped_blur_bw = threshold_hasan_karan(255 - cropped_blur, 1)
    if cropped_blur_bw is None:
        print 'Empty'
        return
    
    area = cropped_blur_bw.sum() / 255.0
    extent = cropped_blur_bw.sum() / (255.0 * width * height)
        
    # Complexity
    comp = median_complexity(cropped_bw, cropped_blur_bw, 3)
    
    # Binary degree
    bin_index= binary_index(cropped)
    
    # Strokes width degree
    _, var, thickness = chen_stroke_width(cropped_blur_bw)
        
    # Filtering
    remove = 5*[' - ']
    
    # Height filter
    if not ((1.2 * R) <= (y2 - y1) <= (4.8 * R)):
        remove[0] = ' H '
    
    # Complexity filter
    if comp > 0.4:
        remove[1] = ' C '
        
    if bin_index < 0.6 :    
        remove[2] = ' B '
    
    if var > 0.4:
        remove[3] = ' V '
        
    if thickness > 1.2 * R:
        remove[4] = ' T '
                    
    print '%-10d %-10d %-10d %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10s' % \
        (label, width, height, area, extent, comp, bin_index, var, thickness, ''.join([c for c in remove]))
    ax1.imshow(cropped, cmap=plt.get_cmap('gray'))
    ax2.imshow(cropped_blur, cmap=plt.get_cmap('gray'))
    ax3.imshow(grad[y1:y2, x1:x2], cmap=plt.get_cmap('gray'))
    ax4.imshow(cropped_bw, cmap=plt.get_cmap('gray'))
    ax5.imshow(cropped_blur_bw, cmap=plt.get_cmap('gray'))
    ax6.imshow(labels[y1:y2, x1:x2])
    fig.canvas.draw()

def press(event):
    
    if event.key in ['left', 'right', 'down', 'up']:
        if event.key == 'left':
            plot(iprops.previous())
        elif event.key == 'right':
            plot(iprops.next())
        elif event.key == 'up':
            plot(iprops.go(0))
        else:
            plot(iprops.go(len(props) - 1))
#    elif event.key == 'g':
#        plot(iprops.go(int(raw_input('pos: '))))
                  
# Open test image
assert len(sys.argv) > 2
img_filename = sys.argv[1]
dpi = int(sys.argv[2])
R = dpi / 25.4

fig = plt.figure(figsize=(2, 2), dpi=dpi)
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax6.axis('off')

# Open as graysacle
img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)

# Blur image
blur = blur_hasan_karan(img)
_, bw = threshold_hasan_karan(255 - blur, 1)

# Gradient
h, w = img.shape
s_33 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
grad = cv2.morphologyEx(255 - blur, cv2.MORPH_GRADIENT, s_33)
#bw = cv2.adaptiveThreshold(
#    grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0
#)
_, bw_grad = threshold_hasan_karan(grad, 1)

dx = int(2 * 1.2 * R / 2)
dy = int(2 * 1.2 * R / 4)
s_xy = cv2.getStructuringElement(cv2.MORPH_RECT, (dx, dy))
es = cv2.morphologyEx(bw_grad, cv2.MORPH_CLOSE, s_xy)

labels = forward_backward_propagation(es)
props = measure.regionprops(labels)
iprops = fiter(props)
fig.canvas.mpl_connect('key_press_event', press)
print 'id         width      height     area       extent     complexity' + \
      ' bin       var        thickness false pos.'
try:
    plot(iprops.next())
    plt.show()
except IndexError:
    print 'Empty page'