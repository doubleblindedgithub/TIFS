import numpy as np
import cv2
import matplotlib.patches as patches
from scipy.spatial.distance import squareform
from stripstext import StripsText
from ..image.utils import join
from ..image.metrics.distance import modified_hausdorff
import matplotlib.pyplot as plt


class StripsChar(StripsText):
    ''' Strips characters class.'''

    def __init__(self, **kwargs):
      
        # Parent constructor
        super(StripsChar, self).__init__(**kwargs)
    
        # Inner characters
        self.inner = []
        # All characters (including joined)
        self.all = []
        # Matchings (right char. of strip i, left char. of strip j, joined)
        self.matchings = {}
        
        filter_blanks = False
        if 'filter_blanks' in kwargs:
            filter_blanks = kwargs['filter_blanks']

        if len(self.strips) > 0:
            self._extract_characters()
            if filter_blanks:
                has_edge_chars = lambda strip: \
                    (len(strip.left) + len(strip.right)) > 0
                self.strips = filter(has_edge_chars, self.strips)
        
        self._compute_matchings()
        self._compute_all_characters()
            
    
    def _extract_characters(self, d_perc=0.5, h_perc=0.6, a_perc=0.4):
        ''' Extract characters from strips. '''
        
        # Displacement from border
        d = int(d_perc * self.R)
        # Height range
        min_height = self.min_height
        max_height = 2.0 / 3.0 * self.max_height 
        # Extent range
        max_extent = 0.7
        # extraction
        inner = []
        for strip in self.strips:
            strip.extract_characters(d, min_height, max_height, max_extent)
            inner += strip.inner

        # Outliers removal for inner characters (area / height)
        # Filter by width / height
        boxes, _ = zip(*inner)
        values = np.array(
            [[box[2], box[3], box[2] * box[3]] for box in boxes],
            dtype=np.float32
        )
        h_med, w_med, a_med = np.median(values, axis=0)
        for strip in self.strips:
            strip.inner = [(box, patch) for box, patch in strip.inner
               if box[3] > h_perc * h_med and \
                   box[2] * box[3] > a_perc * a_med
            ]
#            strip.inner = [(box, patch) for box, patch in strip.inner
#               if box[3] > h_perc * h_med and \
#                   box[2] * box[3] > (a_perc * a_med) and (box[1] < 900 or box[1] > 2980)
#            ]
#            strip.left = [(box, patch) for box, patch in strip.left
#               if (box[1] < 900 or box[1] > 2980)
#            ]
#            strip.right = [(box, patch) for box, patch in strip.right
#               if (box[1] < 900 or box[1] > 2980)
#            ]
        
        self.inner = reduce(
            lambda x, y: x + y, [strip.inner for strip in self.strips]
        )
        

    def _compute_matchings(self, min_overlap=0.5):
        ''' Create matching images by mergind edge characters. '''
        
        assert len(self.inner) > 0
        
        not_null = lambda image: cv2.findContours(
            image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )[1] != []
        
        # For each i = 0, ..., n - 1, j = 0, ..., n - 1, i != j
        self.matchings.clear()
        n = len(self.strips)
        for i in range(n):
            right = self.strips[i].right
            for j in range(i) + range(i + 1, n):
                left = self.strips[j].left

                # Check matching
                matchings = []
                r = 0
                l = 0
                while r < len(right) and l < len(left):
                    boxr, charr = right[r]
                    boxl, charl = left[l]
                    xr, yr, wr, hr = boxr
                    xl, yl, wl, hl = boxl

                    # Check vertical overlapping
                    intersection = min(yr + hr - 1, yl + hl - 1) - max(yr, yl)
                    overlap = float(intersection) / min(hr, hl)
                    if overlap >= min_overlap:
                        m = min(yr, yl)
                        # Contour extraction
                        joined = join(charr, charl, yr - m, yl - m, cval=0)
                        if not_null(joined):
                            matchings.append((charr, charl, joined))
                        l += 1
                        r += 1
                    elif yr + hr >= yl + hl:
                        matchings.append((None, charl, charl))
                        l += 1
                    else:
                        matchings.append((charr, None, charr))
                        r += 1
                            
                while r < len(right):
                    _, charr = right[r]
                    matchings.append((charr, None, charr))
                    r += 1
                    
                while l < len(left):
                    _, charl = left[l]
                    matchings.append((None, charl, charl))
                    l += 1

                self.matchings[(i, j)] = matchings
        
        
    def _compute_all_characters(self):
        
        left = reduce(
            lambda x, y: x + y, [strip.left for strip in self.strips]
        )
        right = reduce(
            lambda x, y: x + y, [strip.right for strip in self.strips]
        )
        joined = reduce(
            lambda x, y: x + y,
            [matchings for matchings in self.matchings.values()]
        )
        _, inner = zip(*self.inner)
        _, left = zip(*left)
        _, right = zip(*right)
        _, _, joined = zip(*joined)
        self.all = inner + left + right + joined

    
    def trim(self, left, right):
        ''' Trim strips. '''
        
        super(StripsChar, self).trim(left, right)
        self._compute_matchings()
        self._compute_all_characters()
    
    
    def plot(self, **kwargs):
        ''' Show background image and text boxes. '''

        fig, ax, offsets = super(StripsChar, self).plot(**kwargs)
        for strip, offset in zip(self.strips, offsets):
            for box, _ in strip.left:
                ax.add_patch(
                    patches.Rectangle(
                        (box[0] + offset, box[1]), box[2], box[3],
                        facecolor='green', edgecolor='none', alpha=0.5
                    )
                )
            for box, _ in strip.right:
                ax.add_patch(
                    patches.Rectangle(
                        (box[0] + offset, box[1]), box[2], box[3],
                        facecolor='red', edgecolor='none', alpha=0.5
                    )
                )
        
            # Inner characters
            for box, _ in strip.inner:
                ax.add_patch(
                    patches.Rectangle(
                        (box[0] + offset, box[1]), box[2], box[3],
                        facecolor='blue', edgecolor='none', alpha=0.4
                    )
                )
        return fig, ax, offsets