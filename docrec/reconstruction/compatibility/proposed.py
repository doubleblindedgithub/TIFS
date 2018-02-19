import numpy as np
from scipy.spatial.distance import squareform
from time import clock
import sys
from algorithm import Algorithm
from docrec.image.processing.transform import distance
from docrec.clustering.kmedoids.kmedoids import KMedoids


class Proposed(Algorithm):
    '''  Proposed algorithm. '''

    def __init__(
        self, strips, max_cache_size=10000000, seed=None, verbose=True,
        trailing=0
    ):
        
        self.strips = strips
        self.verbose = verbose
        self.trailing = trailing
        
        # Distance transform cache structure
        self.cache_distance_transform = {}
        
        # Cache configuration
        self.max_cache_size = max_cache_size
        self.cache_distances = {}
        if verbose:
            print '%sCaching distance transform... ' % (self.trailing * ' '), 
            sys.stdout.flush()
            t0 = clock()
        self._cache_distance_transform()
        if verbose:
            print 'Elapsed time: %.2fs' % (clock() - t0)
            sys.stdout.flush()            
        
        self.inner_distances = None
        if verbose:
            print '%sComputing inner distances... ' % (self.trailing * ' '),
            sys.stdout.flush()            
            t0 = clock()
        self._compute_inner_distances()
        if verbose:
            print 'Elapsed time: %.2fs' % (clock() - t0)
            sys.stdout.flush()            

        # Random number generator
        self.rng = np.random.RandomState(seed)
            
        # Results
        self.representative = None
        self.matrix = None
        
        # Parameters
        self.FFm = None
        self.EC = None
        self.FF = None
        self.CC = None
        self.EF = None
        self.gamma = None
        self.threshold = None
        
    
    def _cache_distance_transform(self):
        ''' Caching distance transform. '''
        
        max_w = max([char.shape[1] for char in self.strips.all])
        max_h = max([char.shape[0] for char in self.strips.all])
        
        for char in self.strips.all:
            id_ = id(char)
            self.cache_distance_transform[id_] = distance(
                char, window=(2 * max_w, 2 * max_h)
            )


    def _distance(self, char1, char2):
        ''' Distance (dissimilarity) between two characters. '''
        
        id1 = id(char1)
        id2 = id(char2)
        try:
            dist = self.cache_distances[(id1, id2)]
        except KeyError:
            distance_transform1, base1 = self.cache_distance_transform[id1]
            distance_transform2, base2 = self.cache_distance_transform[id2]
            h12 = distance_transform1[base2 == 255].mean()
            h21 = distance_transform2[base1 == 255].mean()
            dist = max(h12, h21)
            if len(self.cache_distances) < self.max_cache_size:
                self.cache_distances[(id1, id2)] = dist
        return dist

    
    def _compute_inner_distances(self):
        ''' Inner characters distantes for clustering. '''
        
        _, chars = zip(*self.strips.inner)
        
        # Compute distances
        dists = [self._distance(char1, char2)
            for i, char1 in enumerate(chars[: -1])
            for char2 in chars[i + 1 :]
        ]
        self.inner_distances = squareform(dists)
        
    
    def _is_character(self, char):
        ''' Search for any similar character. '''
    
        for other in self.representative:
            if self._distance(char, other) <= self.threshold:
                return True
        return False


    def _score_tresh(self, charr, charl, joined):
        ''' Threshold-based score function. '''
        
        # EC EF
        if charr is None:
            if self._is_character(charl):
                return self.EC
            return self.EF
        elif charl is None:
            if self._is_character(charr):
                return self.EC
            return self.EF

        # FFm
        if self._is_character(joined):
            return self.FFm

        # CC CF FC FF
        rtype = 'C' if self._is_character(charr) else 'F'
        ltype = 'C' if self._is_character(charl) else 'F'
        mtype =  rtype + ltype
        if mtype == 'CC':
            return self.CC
        if mtype == 'FF':
            return self.FF
        return self.FC
       
    
    def _compute_score(self, seed):
        ''' Compute pairwise score. '''
                
        # Clustering
        # Obs: parameter k value not used until this moment.
        X = self.inner_distances
        n = X.shape[0]
        #k = n / 4
        # ISO basic Latin alphabet
        k = min(n, 52)
        max_neighbor = int(self.gamma * n * (n - k))
        kmedoids = KMedoids(
            seed=seed, init='random', num_local=2, max_neighbor=max_neighbor
        ).run(X, k)
        
        # Representative characters
        _, inner = zip(*self.strips.inner)
        self.representative = [inner[i] for i in kmedoids.medoids]
        score_func = self._score_tresh
        score = {}
        for pair in self.strips.matchings:
            val = 0
            for charr, charl, joined in self.strips.matchings[pair]:
                val += score_func(charr, charl, joined)
            score[pair] = val
        return score


    def _compute_matrix(self, seed):
        ''' Compute matrix. '''
        
        assert self.inner_distances is not None
        assert self.FFm is not None
        assert self.EC is not None
        assert self.FF is not None
        assert self.CC is not None
        assert self.EF is not None
        assert self.gamma is not None
        assert  self.threshold is not None
        
        # Score computation
        score = self._compute_score(seed)
        
        # Filling cost matrix
        n = len(self.strips.strips)
        matrix = np.zeros((n, n), dtype=np.float32)
        for pair in score:
            matrix[pair] = score[pair]

        # Transformation function
        matrix = matrix.max() - matrix
        np.fill_diagonal(matrix, np.inf)
        return matrix


    def run(
        self, FFm=1.0, CC=1.0, EC=0.0, p=-0.2, gamma=0.0125,
        threshold=1.0, ns=10
    ):
        ''' Run algorithm. '''
        
        # Parameters
        self.CC = CC
        self.EF = self.FC = self.FF = p
        self.FFm = FFm
        self.EC = EC
        self.gamma = gamma
        self.threshold = threshold
    
        verbose = self.verbose
    
        # Matrix computation
        matrices = []
        for s in range(1, ns + 1):
            if verbose:
                print '%ssolution %d - (cache size = %d)... ' %  \
                    (self.trailing * ' ', s, len(self.cache_distances)),
                sys.stdout.flush()
                t0 = clock()
            # Seed for clustering
            matrix = self._compute_matrix(self.rng.randint(0, 4294967295))
            matrices.append(matrix)
            if verbose:
                print 'Elapsed time: %.2fs' % (clock() - t0)
                sys.stdout.flush()
        
        self.matrix = np.stack(matrices)
        return self


    def name(self):
        ''' Method name. '''
        
        return 'proposed'
