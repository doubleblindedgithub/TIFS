import numpy as np
from algorithm import Algorithm
import libjigsaw


class Andalo(Algorithm):
    ''' Algorithm Andalo 2017. '''

    def __init__(self, strips):
      
        # Instance
        self.strips = strips
        
        # Result 
        self.matrix = None

   
    def _compute_matrix(self, p, q, d):
        ''' Compute cost matrix. '''

        matrix = libjigsaw.compatibility(
            self.features(d), len(self.strips.strips), p, q
        )
        np.fill_diagonal(matrix, np.inf)
        return matrix


    def features(self, d=0):
        ''' Features. '''
    
        min_h = min([strip.h for strip in self.strips.strips])
        first = lambda strip: strip.left_borders_coordinates()[: min_h]
        last = lambda strip: strip.right_borders_coordinates()[: min_h]

        L = map(first, self.strips.strips)
        R = map(last, self.strips.strips)
        
        # Extract borders
        idx = np.arange(min_h)
        features = []
        temp = np.zeros((min_h, 4 , 3), dtype=np.uint8)
        for l, r, strip in zip(L, R, self.strips.strips):
            temp[:, 0, : ] = strip.image[idx, l + d, :]
            temp[:, 1, : ] = strip.image[idx, l + d + 1, :]
            temp[:, 2, : ] = strip.image[idx, r - d - 1, :]
            temp[:, 3, : ] = strip.image[idx, r - d, :]
            features.append(temp.copy())
            temp[:] = 0
        
        stacked = np.hstack(features)
        return stacked
    
    def run(self, p=1.0, q=0.3, d=0):
        ''' Run algorithm. '''
        
        self.matrix = self._compute_matrix(p, q, d)
        return self
    
    def name(self):
        
        return 'andalo'
