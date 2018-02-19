import sys
sys.path.append('../../')
import os
import cPickle as pickle
import matplotlib.pyplot as plt
from docrec.validation.config.dataset import DatasetConfig
from docrec.strips.strips import Strips

assert len(sys.argv) == 2

filename = 'dataset.cfg'
id = sys.argv[1]

curr = os.getcwd()
os.chdir('../../')
dataset_config = DatasetConfig(filename)
datasets = dataset_config.datasets
dataset = None
for candidate in datasets:
    if candidate.id == id:
        dataset = candidate
        break

docs = dataset.docs
margins = {doc.id : {'left' : 0, 'right' : 0} for doc in docs}

cnt = [0]
sides = ['left', 'right']
cnt_side = [0]

# Figure
dpi = 80
margin = 0.05 # (5% of the width/height of the figure...)
xpixels, ypixels = 800, 800
figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi
fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
ax.axis('off')


def plot():
    doc = docs[cnt[0]]
    ax.clear()
    strips = Strips(path=doc.path, filter_blanks=False)
    ax.set_title(doc.id)
    strips.plot(ax=ax, show_lines=True)
    fig.canvas.draw()


def press(event):
    
    if event.key == 'w': # save
        print 'write'
        pickle.dump(
            margins, open('margins_%s.pkl'  % dataset, 'w')
        )
    elif event.key == 'q': # quit
        print 'quit'    
        os.chdir(curr)
        sys.exit()
    elif event.key == 'right':
        cnt[0] = min(cnt[0] + 1, len(docs) - 1)
        cnt_side[0] = 0
        plot()
    
    elif event.key == 'left':
        cnt[0] = max(cnt[0] - 1, 0)
        cnt_side[0] = 0
        plot()
    elif event.key in [str(i) for i in range(10)]:
        side = sides[cnt_side[0]]
        cnt_side[0] = (cnt_side[0] + 1) % 2
        doc = docs[cnt[0]]
        margins[doc.id][side] = int(event.key)
        print '%s: left=%d right=%d' % \
            (doc.id, margins[doc.id]['left'], margins[doc.id]['right'])
        
fig.canvas.mpl_connect('key_press_event', press)

plot()
plt.show()    
