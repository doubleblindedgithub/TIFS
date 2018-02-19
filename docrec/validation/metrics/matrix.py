import numpy as np
from scipy import stats

def _average_rank_2D(matrix):
    ''' Average rank of a single 2D matrix.'''
            
    rank_rows = np.apply_along_axis(
        stats.rankdata, 1, matrix
    ).diagonal(offset=1)
    rank_cols = np.apply_along_axis(
        stats.rankdata, 0, matrix
    ).diagonal(offset=1)

    return 0.5 * (rank_rows + rank_cols).mean()


def average_rank(matrix):
    ''' Average rank of rows and columns. (1, n - 1)'''
        
    assert matrix.ndim in (2, 3)
    
    if matrix.ndim == 2:
        return [_average_rank_2D(matrix)]
    else:
        return [_average_rank_2D(matrix_) for matrix_ in matrix]
        

def normalized_average_rank(matrix):
    ''' Normalized average rank of rows and columns (0, 1). '''
        
    assert matrix.ndim in (2, 3)
    
    n = matrix.shape[1]
    if matrix.ndim == 2:    
        avr = average_rank(matrix)
        return (avr - 1.0) / (n - 2.0)
    else:
        avr = np.array(average_rank(matrix))
        return ((avr - 1.0) / (n - 2.0)).tolist()
    
 
def _number_of_perfect_matchings(matrix, axis):
    ''' Number of perfect matchings.'''
            
    rank = np.apply_along_axis(stats.rankdata, axis, matrix).diagonal(offset=1)
    #.diagonal(offset=1)
    #print rank

    return np.sum(rank == 1)


#def _number_of_perfect_matchings(matrix, axis):
#    
#    n = matrix.shape[0]
#    values = matrix.diagonal(offset=1)
#    min_values = matrix.min(axis=axis)[1 - axis : n - axis]
#    
#    return np.sum(values == min_values)

    
def prediction_eff(matrix, method='fix_left'):

    assert matrix.ndim in (2, 3)
    assert method in ('fix_left', 'fix_right')
    
    n = matrix.shape[1]
    axis = 0 if method == 'fix_right' else 1
    matching_func = lambda matrix : _number_of_perfect_matchings(
        matrix, axis
    ) / (n - 1.0)
    if matrix.ndim == 2:
        return matching_func(matrix)
    else:
        return np.array([matching_func(matrix_) for matrix_ in matrix])