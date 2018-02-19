import numpy as np

def check_matrix(matrix, max_null=0.4):
    ''' Check matrix validity. '''
    
    matrix_ = matrix.copy()
    np.fill_diagonal(matrix_, 0)
    null = 0
    for row in matrix_:
        if (row - row.min()).sum() == 0:
            null += 1
         
    return (float(null) / matrix_.shape[0]) < max_null