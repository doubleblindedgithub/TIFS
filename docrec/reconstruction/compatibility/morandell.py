import numpy as np
from numba import jit
import cv2
from algorithm import Algorithm


@jit(nopython=True)
def _delta_x(X, Y, X_dist, Y_closest, epsilon):
    ''' Calculate delta_x values. '''

    for x in xrange(epsilon, X.size - epsilon):
        X_dist[x - epsilon] = epsilon + 1

        if not X[x]:
            X_dist[x - epsilon] = 0
            continue

        # Check zero distance
        if Y[x]:
            X_dist[x - epsilon] = 0
            Y_closest[x - epsilon] = 1
            continue

        # Check distances less or equal then epsilon
        for d in xrange(1, epsilon + 1):
            if Y[x - d]:
                X_dist[x - epsilon] = d
                Y_closest[x - d - epsilon] = 1
                break

            if Y[x + d]:
                X_dist[x - epsilon] = d
                Y_closest[x + d - epsilon] = 1
                break

@jit(nopython=True)
def _delta_y(Y, X, Y_dist, epsilon):
    ''' Calculate delta_x values. '''

    for y in xrange(epsilon, Y.size - epsilon):
        Y_dist[y - epsilon] = epsilon + 1

        if not Y[y]:
            Y_dist[y - epsilon] = 0
            continue

        # Check zero distance
        if X[y]:
            Y_dist[y - epsilon] = 0
            continue

        # Check distances less or equal then epsilon
        for d in xrange(1, epsilon + 1):
            if X[y - d] or X[y + d]:
                Y_dist[y - epsilon] = d
                break


class Morandell(Algorithm):
    '''
    Algorithm Morandell

    Evaluation and Reconstruction of Strip-Shredded Text Documents (2008).
    '''

    def __init__(self, strips):
        ''' Constructor. '''

        # Instance
        self.strips = strips
        
        # Result 
        self.matrix = None


    def _cost(self, X, Y, epsilon, h, pi, phi):

        # Invert and pad image
        X_ext = np.pad(X, (epsilon, epsilon), mode='constant', constant_values=False)
        Y_ext = np.pad(Y, (epsilon, epsilon), mode='constant', constant_values=False)

        X_dist = np.empty(X.size, dtype=np.int32)
        Y_dist = np.empty(Y.size, dtype=np.int32)

        # X -> Y cost
        Y_closest = np.zeros(Y.size, dtype=np.uint8)
        _delta_x(X_ext, Y_ext, X_dist, Y_closest, epsilon)

        # Exclude Y* pixels from analysis
        Y_star = np.where(Y_closest == 1)[0]
        Y_ext[Y_star + epsilon] = False

        # Y -> X cost
        _delta_y(Y_ext, X_ext, Y_dist, epsilon)

        correction = np.vectorize(lambda d: pi if d == 0 else (d**h if d <= epsilon else epsilon**h + phi))
        X_dist = np.apply_along_axis(correction, 0, X_dist)
        Y_dist = np.apply_along_axis(correction, 0, Y_dist)

        return X_dist.sum() + Y_dist.sum()


    def _compute_matrix(self, d, epsilon, h, pi, phi):
        ''' Compute matrix. '''

        l, r = self.features(d)
        
        # Distance computation
        dist = lambda x, y : self._cost(x, y, epsilon, h, pi, phi)
        
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
    
    
    def run(self, d=0, epsilon=10, h=1, pi=0, phi=0):
        
        self.matrix = self._compute_matrix(d, epsilon, h, pi, phi)
        return self

    def name(self):
        return 'Morandell'
