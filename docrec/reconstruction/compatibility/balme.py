import numpy as np
import cv2
from algorithm import Algorithm


class Balme(Algorithm):
    '''
    Algorithm Balme

    Balme, J.: Reconstruction of shredded documents in the absence of shape
    information (2007) Working paper, Dept. of Computer Science, Yale
    University, USA.
    '''

    def __init__(self, strips):

        # Instance
        self.strips = strips
        
        # Result 
        self.matrix = None
        

    def _compute_matrix(self, d, tau):
        ''' Compute cost matrix. '''

        l, r = self.features(d)
        
        # Distance computation
        # Gaussian correlation
        dist = lambda x, y: np.sum(
            np.correlate(
                np.logical_xor(x, y), [0.05, 0.1, 0.7, 0.1, 0.05]
            ) > tau
        )

        n = len(self.strips.strips)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = dist(r[i], l[j])

        np.fill_diagonal(matrix, np.inf)
        return matrix
    
    
    def features(self, d=0):
        ''' Features. '''
    
        # Inverted thresholded image
        convert = lambda image : \
            cv2.threshold(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1] == 0
        images = [convert(strip.image) for strip in self.strips.strips]
                
        min_h = min([strip.h for strip in self.strips.strips])
        first = lambda strip: strip.left_borders_coordinates()[: min_h]
        last = lambda strip: strip.right_borders_coordinates()[: min_h]

        L = map(first, self.strips.strips)
        R = map(last, self.strips.strips)
        
        # Extract borders
        idx = np.arange(min_h)
        left_features = []
        right_features = []
        for l, r, image in zip(L, R, images):
            left_features.append(image[idx, l + d])
            right_features.append(image[idx, r - d])
        return left_features, right_features
    
    
    def run(self, d=0, tau=0.1):
        
        self.matrix = self._compute_matrix(d, tau)
        return self
    
    
    def name(self):
        return 'balme'
