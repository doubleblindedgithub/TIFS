import numpy as np
import cv2
import copy
from ..ndarray.utils import first_nonzero, last_nonzero
from ..ocr.text.extraction import extract_text
from ..ocr.character.extraction import extract_characters


class Strip(object):
    ''' Strip image.'''
    
    def __init__(self, image, position, mask=None):

        h, w = image.shape[: 2]
        if mask is None:
            mask = 255 * np.ones((h, w), dtype=np.uint8)
        
        self.h = h
        self.w = w
        self.image = cv2.bitwise_and(image, image, mask=mask)
        self.position = position
        self.mask = mask
        
        # [(box = (x, y, w, h), patch), ...]
        self.text = []
        
        # Characters
        self.inner = []
        self.left = []
        self.right = []
    
    
    def copy(self):
        ''' Copy object. '''
        
        return copy.deepcopy(self)
    
    
    def filled_image(self):
        ''' Return image with masked-out areas in white. '''
        
        return cv2.bitwise_or(
            self.image, cv2.cvtColor(
                cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2RGB
            )
        )
    
    
    def is_blank(self, blank_tresh=127):
        ''' Check if is a blank strip. '''
        
        blurred = cv2.GaussianBlur(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        return (blurred < blank_tresh).sum() == 0
    
    
    def approximate_width(self):
        ''' Aproximate width. '''
        
        l = self.left_borders_coordinates()
        r = self.right_borders_coordinates()
        return int(np.mean(r - l + 1))
    
    
    def left_borders_coordinates(self):
        ''' Left borders coordinates. '''
        
        return np.apply_along_axis(first_nonzero, 1, self.mask)


    def right_borders_coordinates(self):
        ''' Right borders coordinates. '''
        
        return np.apply_along_axis(last_nonzero, 1, self.mask)
    
    
    def stack(self, other):
        ''' Stack horinzontally with other strip. '''
        
        min_h, max_h = min(self.h, other.h), max(self.h, other.h)
        
        # Borders coordinates
        r1 = self.right_borders_coordinates()[: min_h]
        l2 = other.left_borders_coordinates()[: min_h]
        
        # Offset
        disp = np.min(l2 + self.w - 1 - r1)
        offset = self.w - disp
        
        temp_image = np.zeros((max_h, offset + other.w, 3), dtype=np.uint8)
        temp_image[: self.h, : self.w, :] = self.image
        temp_image[: other.h, offset :, : ] += other.image
        
        temp_mask = np.zeros((max_h, offset + other.w), dtype=np.uint8)
        temp_mask[: self.h, : self.w] = self.mask
        temp_mask[: other.h, offset :] += other.mask
        
        self.h, self.w = temp_mask.shape
        self.image = temp_image
        self.mask = temp_mask
        
        return offset
   
    
    def extract_text(self, min_height, max_height, max_separation):
        '''Extract text information contained in the strip. '''
        
        self.text = extract_text(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY),
            min_height, max_height, max_separation, 0.95
        )
        
    
    def extract_characters(self, d, min_height, max_height, max_extent):
        ''' Extract characters information. '''
        
        # Extent range
        left = []
        right = []
        inner = []
        if self.text:
            # Borders coordinates
            lb = self.left_borders_coordinates()
            rb = self.right_borders_coordinates()
            lb += d
            rb -= d
            # Character extraction
            chars = []
            for box, patch in self.text:
                x, y, w, h = box
                chars += extract_characters(
                    patch, offset=(x, y),
                    max_width=self.approximate_width() / 2
                )
                
            # Categorization
            for char in chars:
                box, patch = char
                x, y, w, h = box
                
                # Left ?
                if np.any(x <= lb[y : y + h]):
#                    if 0.5 * min_height <= h <= max_height:
                        left.append(char)
                # Right ?
                elif np.any(x + w - 1 >= rb[y : y + h]):
#                    if 0.5 * min_height <= h <= max_height:
                        right.append(char)
                # Inner !
                else:
#                    extent = float((patch == 255).sum()) / (w * h)
#                    if (min_height <= h <= max_height) and \
#                        (extent < max_extent):
                        inner.append(char)
        
        filtered_inner = []
        filtered_inner = inner
#        for char in inner:
#            box, patch = char
#            x, y, w, h = box
#            if w >= 2 and h >= 2:
#                props = measure.regionprops(measure.label(patch))
#                if len(props) == 1:
#                    filtered_inner.append(char)
#                else:
#                    largest = props[np.argmax([region.area for region in props])]
#                    yc, xc, _, _ = largest.bbox
#                    hc, wc = largest.image.shape
#                    box = (xc + x, yc + y, wc, hc)
#                    patch = 255 * (largest.image.astype(np.uint8))
#                    filtered_inner.append((box, patch))
#        
        # Filtering non-null characters (OpenCV findContour requirements)
        not_null = lambda char: cv2.findContours(
            char[1].copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )[1] != []
        self.inner = filter(not_null, filtered_inner)
        self.left = filter(not_null, left)
        self.right = filter(not_null, right)
