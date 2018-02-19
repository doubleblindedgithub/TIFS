import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2
from docrec.ocr.text.features import chen_stroke_width
from docrec.image.processing.threshold import threshold_hasan_karan
from docrec.image.processing.label import forward_backward_propagation
from docrec.image.processing.blur import blur_hasan_karan
from docrec.image.metrics.complexity import median_complexity, binary_index
from docrec.strips.stripstext import StripsText
import matplotlib.patches as patches

#strips = StripsText(path='../dataset/andalo_2017/mechanical/D001')
strips = StripsText(path='../dataset/marques_2013/D057')
strip = [0]
text = [0]

def plot(dpi=300):
    sys.stdout.flush()
    image = strips(strip[0]).image
    mask = strips(strip[0]).mask
    box, bw = strips(strip[0]).text[text[0]]
    x, y, w, h = box
    R = dpi / 25.44
#    
#    cropped = img[y1:y2, x1:x2]
#    _, cropped_bw = threshold_hasan_karan(255 - cropped, 1)
#    if cropped_bw is None:
#        print 'Empty'
#        return
#    
#    cropped_blur = blur_hasan_karan(cropped)
#    _, cropped_blur_bw = threshold_hasan_karan(255 - cropped_blur, 1)
#    if cropped_blur_bw is None:
#        print 'Empty'
#        return
#    
#    area = cropped_blur_bw.sum() / 255.0
#    extent = cropped_blur_bw.sum() / (255.0 * width * height)
#        
    # Complexity
#    comp = median_complexity(cropped_bw, cropped_blur_bw, 3)
#    
    # Binary degree
    gray = cv2.cvtColor(image[y : y + h, x : x + w, :], cv2.COLOR_RGB2GRAY)
    bin_index= binary_index(gray, mask=mask[y : y + h, x : x + w])
#    
    # Strokes width degree
    _, var, thickness = chen_stroke_width(bw)
#        
#    # Filtering
    remove = 5*[' - ']
#    
#    # Height filter
#    if not ((1.2 * R) <= (y2 - y1) <= (4.8 * R)):
#        remove[0] = ' H '
#    
#    # Complexity filter
#    if comp > 0.4:
#        remove[1] = ' C '
#        
    if bin_index < 0.6 :    
        remove[2] = ' B '
#    
#    if var > 0.4:
#        remove[3] = ' V '
#        
    if thickness > 1.2 * R:
        remove[4] = ' T '
                    
    label = area = 0
    
    extent = (bw == 255).sum() / float(w * h)
    
    mask = np.logical_and(mask[y : y + h, x : x + w] == 255, bw == 0)
    mean_color = gray[mask].sum() / float(mask.size)
#    extent = 121
    print '%-10d %-10d %-10d %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10s' % \
        (label, w, h, area, extent, mean_color, bin_index, var, thickness, ''.join([c for c in remove]))

    ax1.clear()
    ax2.clear
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.add_patch(
        patches.Rectangle(
            (x, y), w, h, facecolor='green', edgecolor='none', alpha=0.5)
        )
    ax2.imshow(bw, cmap=plt.cm.gray)
    
    layout = np.zeros_like(image)
    gray = cv2.cvtColor(strips(strip[0]).filled_image(), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v = bw.sum(axis=1) > 20
    layout[v, :] = 255
    ax3.clear()
    ax3.imshow(layout, cmap=plt.cm.gray)
    ax3.axis('off')
    fig.canvas.draw()

def press(event):
    
    if event.key in ['left', 'right', 'down', 'up']:
        if event.key == 'left':
            strip[0] = max(strip[0] - 1, 0)
            text[0] = 0
        elif event.key == 'right':
            strip[0] = min(strip[0] + 1, len(strips.strips) - 1)
            text[0] = 0
        elif event.key == 'up':
            text[0] = max(text[0] - 1, 0)
        else:
            text[0] = min(text[0] + 1, len(strips(strip[0]).text) - 1)
            
        plot()
                  
# Open test image
fig = plt.figure(figsize=(8, 8), dpi=300)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

fig.canvas.mpl_connect('key_press_event', press)
print 'id         width      height     area       extent     mean color' + \
      ' bin       var        thickness false pos.'

plot()
plt.show()
