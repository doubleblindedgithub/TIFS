import numpy as np


def accuracy(solution, method='neighbor'):
    ''' Accuracy by direct  / neighbor comparison. '''
    
    assert len(solution) > 0
    assert method in ('direct', 'neighbor')  
    
    solution = np.array(solution)
    assert solution.ndim in (1, 2)
    
    n = solution.shape[-1]
    if method == 'direct':
        accuracy_func = lambda solution : np.sum(
            solution == np.arange(n)
        ) / float(n)
    else:
        accuracy_func = lambda solution : np.sum(
        np.diff(solution) == 1
    ) / float(n - 1) 
    
    if solution.ndim == 1:
        return accuracy_func(solution)
    else:
        return np.apply_along_axis(accuracy_func, 1, solution)