'''
USAGE python training.py
'''
import sys
import os
import numpy as np
import pandas as pd
from time import clock
from utils import check_matrix
from docrec.strips.stripschar import StripsChar
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path as solve
from docrec.reconstruction.compatibility.proposed import Proposed
from docrec.validation.config.experiments import ExperimentsConfig

# Global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# Resulting dataframe
index = index_backup = 0
df = pd.DataFrame(
    columns=('document', 'threshold', 'run', 'accuracy', 'solution')
)
df_filename = config.path_cache('training.csv')
if os.path.exists(df_filename):
    index_backup = len(pd.read_csv(df_filename, sep='\t'))
else:
    df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)

dataset = config.dataset_config.datasets['training'] 
docs = dataset.docs

params = config.algorithms_config.algorithms['proposed'].params
params['ns'] = 1
thresholds = config.thresholds

ndocs = len(docs)
nruns = config.nruns
nthresh = len(thresholds)
total = ndocs * nthresh * nruns

for doc in docs:

    # Skip document
    if index + nthresh * nruns <= index_backup:
        index += nthresh * nruns
        continue
    
    print 'Segmentating characters of %s ... ' % doc.id,
    sys.stdout.flush()  
    t0 = clock()
    strips = StripsChar(path=doc.path, filter_blanks=True)
    print 'Elapsed time: %.2fs' % (clock() - t0)
    sys.stdout.flush()  
    
    print 'Building algorithm object for %s ... ' % doc.id
    sys.stdout.flush()  
    t0 = clock()
    seed = (hash(doc.id) + config.seed) % 4294967295
    alg = Proposed(strips, seed=seed, verbose=True, trailing=8)
    print 'Elapsed time: %.2fs' % (clock() - t0)
    sys.stdout.flush() 
    
    for t, threshold in enumerate(thresholds):
        
        # Skip threshold
        if index + nruns <= index_backup:
            index += nruns
            continue
        
        params['threshold'] = threshold
        for run in range(1, nruns + 1):
            
            # Skip run
            if index + 1 <= index_backup:
                index += 1
                continue
            
            done = 100 * float(index + 1) / total
            print '%s[%.2f%%] doc=%s threshold=%.2f run=%d' % \
                (4 * ' ', done, doc.id, threshold, run)
            sys.stdout.flush() 
            
            print '%sMatrix ... ' % (4 * ' ')
            sys.stdout.flush() 
            t0 = clock()
            matrix = alg(**params).matrix[0]
            print '%sElapsed time: %.2fs' % (4 * ' ', clock() - t0)
            sys.stdout.flush() 
            
            print '%sSolution ... ' % (4 * ' '),
            sys.stdout.flush() 
            t0 = clock()
            acc = float('nan')
            sol = None
            if check_matrix(matrix):
                sol, _ = solve(matrix)
                if sol is not None:
                    acc = accuracy(sol)
            sol = ' '.join(str(v) for v in sol) if sol is not None else ''
            print '%sElapsed time: %.2fs' % (4 * ' ', clock() - t0)
            sys.stdout.flush() 
            
            # Dumping
            df.loc[index] = [doc.id, threshold, run, acc, sol]
            df.to_csv(
                df_filename, sep='\t', encoding='utf-8', index=False,
                mode='a', header=False
            )
            df.drop(df.index, inplace=True) # clear
            index += 1
            
# Search best threshold
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')            
df = df[~ np.isnan(df.accuracy)]
keys = ['threshold']
print 'Best threshold: %.2f' %  df.groupby(keys)['accuracy'].mean().idxmax()

