import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

fig, ax = plt.subplots(figsize=(5, 10))

# Path to dataset
assert len(sys.argv) > 1

doc = sys.argv[1]
src_path = 'dataset/D1/mechanical_'
tgt_path = 'dataset/D1/mechanical'
full_images_path = 'dataset/D1/full_images'

if os.path.exists(tgt_path):
    shutil.rmtree(tgt_path)

src = os.path.join(src_path, doc, '%s.jpg' % doc)
dst = os.path.join(full_images_path, '%s.jpg' % doc)
shutil.copyfile(src, dst)

os.makedirs(tgt_path)
os.makedirs(os.path.join(tgt_path, doc, 'strips'))

files = sorted(os.listdir(os.path.join(src_path, doc, 'strips')))
strips = []
for file_ in files:
    strip = cv2.imread(os.path.join(src_path, doc, 'strips', file_))
    strips.append(strip)

n_strips = len(strips) 
image = [None]
cnt = [0]
cuts = n_strips * [-1]

def update_image():
    
    print 'Processing %s%02d.jpg' % (doc, cnt[0])
    sys.stdout.flush()
    
    image[0] = strips[cnt[0]][ :,:,:]

def save():
    
    left = np.zeros((strips[0].shape[0], 0, 3), np.uint8)
    for i, strip in enumerate(strips):
        filename = os.path.join(
            tgt_path, doc, 'strips', '%s%02d.jpg' % (doc, i + 1)
        )
        w = strip.shape[1]
        cut = w if cuts[i] == -1 else cuts[i]
        right = strip[:, : cut, :]
        cv2.imwrite(filename, np.hstack((left, right)))
        left = strip[:, cut : , :] if cut < w else np.zeros((strips[0].shape[0], 0, 3), np.uint8)

        print filename


def plot():
    
    ax.clear()
    ax.axis('off')
    ax.imshow(image[0], interpolation='nearest')
    ax.set_title('%s%02d.jpg' % (doc, cnt[0]))

    fig.canvas.draw()
#
def press(event):
    if event.key.lower() == 'w':
        save()
    elif event.key == 'm' and event.inaxes is not None:
        cuts[cnt[0]] = int(event.xdata)
        print cuts
        cnt[0] = min(cnt[0] + 1, len(strips) - 1)
        update_image()
        plot()
    elif event.key == 'right':
        cnt[0] = min(cnt[0] + 1, len(strips) - 1)
        update_image()
        plot()
    elif event.key == 'left':
        cnt[0] = max(cnt[0] - 1, 0)
        update_image()
        plot()

fig.canvas.mpl_connect('key_press_event', press)
update_image()
plot()
plt.show()
