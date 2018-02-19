import cv2
import numpy as np
from algorithm import Algorithm


class Marques(Algorithm):
    ''' Algorithm Marques 2013. '''

    def __init__(self, strips):
      
        # instance
        self.strips = strips
        
        # result 
        self.matrix = None
   
    def _compute_matrix(self, d):
        ''' Compute cost matrix. '''

        l, r = self.features(d)
        
        # distances computation
        dist = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
        
        n = len(self.strips.strips)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = dist(r[i], l[j])
        np.fill_diagonal(matrix, np.inf)
        return matrix

    
    def features(self, d=3):
        ''' Features. '''
    
        # Inverted thresholded image
        convert = lambda image : \
            cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2]
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
    
    
    def run(self, d=3):
        ''' Run algorithm. '''
        
        self.matrix = self._compute_matrix(d)
        
        return self
    
    def name(self):
        
        return 'marques'
