import sys
import os
import cv2
import cPickle as pickle
import matplotlib.pyplot as plt
from docrec.validation.config.dataset import DatasetConfig

'''CATEGORIES

1) Text-only
2) Line-based graphics
3) Full-graphics
'''
id_category =  {'1': 'to', '2': 'lg', '3': 'fg'}
categories = {'to': set([]), 'lg': set([]), 'fg': set([])}

class fiter:
    def __init__(self, seq):
        self.curr = 0
        self.seq = seq

    def pos(self):
        return self.curr
    
    def current(self):
        return self.seq[self.curr]
        
    def previous(self):
        if self.curr > 0:
            self.curr -= 1
        else:
            raise StopIteration
    
    def next(self):
        if self.curr < len(self.seq) - 1:
            self.curr += 1
        else:
            raise StopIteration

def plot():
    doc, docpath = doc_iterator.current()
    image = cv2.imread('%s/%s.jpg' % (docpath, doc), cv2.IMREAD_COLOR)
    ax.imshow(image)
    fig.canvas.draw()


def press(event):
    
    assert event.key in ['1', '2', '3']
    
    try:
        doc, _ = doc_iterator.current()
        category = id_category[event.key]
        print doc_iterator.pos() + 1, doc, category
        categories[category].update([doc])
        doc_iterator.next()
        plot()
    except StopIteration:
        pickle.dump(categories, open('categories_%s.pkl' % sys.argv[1], 'w'))
        sys.exit()    


fig = plt.figure(figsize=(2, 2), dpi=300)
ax = fig.add_subplot(111)
ax.axis('off')
fig.canvas.mpl_connect('key_press_event', press)

assert len(sys.argv) == 2

dataset_config = DatasetConfig('dataset.cfg')
dataset = dataset_config.datasets[sys.argv[1]]
docs = dataset.docs
docspath = [(doc.id, doc.path) for doc in docs]
doc_iterator = fiter(docspath)
plot()
plt.show()
