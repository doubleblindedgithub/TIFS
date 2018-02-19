'''
USAGE python testing_proposed.py
'''
import sys
import os
import numpy as np
from time import clock
import cPickle as pickle
import gc
import pandas as pd
from utils import check_matrix
from docrec.strips.strips import Strips
from docrec.strips.stripschar import StripsChar
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path
from docrec.reconstruction.jigsaw.jigsaw import solve_from_matrix
from docrec.reconstruction.compatibility.proposed import Proposed
from docrec.validation.config.experiments import ExperimentsConfig
from docrec.ocr.recognition import number_of_words


# Global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# Solvers
solve_proposed = lambda mat: shortest_directed_hamiltonian_path(mat)
solve_andalo = lambda mat: solve_from_matrix(mat)
solvers = {'proposed': solve_proposed, 'andalo': solve_andalo}
nsolvers = len(solvers)

# Datasets
datasets = config.dataset_config.datasets
del datasets['training']
ndocs_total = sum([len(dataset.docs) for dataset in datasets.values()])

#==============================================================================
# Proposed system
#==============================================================================

# Resulting dataframe
index = index_backup = 0
df_proposed = pd.DataFrame(
    columns=('solver', 'dataset', 'shredding', 'document', 'run', 's',
             'nwords', 'accuracy', 'cost', 'solution')
)
df_filename = config.path_cache('testing_proposed.csv')
if os.path.exists(df_filename):
    index_backup = len(pd.read_csv(df_filename, sep='\t'))
else:
    df_proposed.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)

params = config.algorithms_config.algorithms['proposed'].params
nruns = config.nruns
ns = params['ns']

total = nsolvers * ndocs_total * nruns * ns
try:
    for dataset in datasets.values():
        
        # Skip dataset
        ndocs = len(dataset.docs)
        if index + ndocs * nsolvers * nruns * ns <= index_backup:
            index += ndocs * nsolvers * nruns * ns
            continue
        
        dset_main_id, dset_shredding = dataset.id.split('-')

        for doc in dataset.docs:
                
            # Skip document
            if index + nsolvers * nruns * ns <= index_backup:
                index += nsolvers * nruns * ns
                continue
            
            print 'Segmentating characters of %s ... ' % doc.id,
            sys.stdout.flush()  
            t0 = clock()
            strips = StripsChar(path=doc.path, filter_blanks=(dset_main_id != 'D2'))
            if dataset.id == 'D2-art':
                margins = pickle.load(open('margins_D2-art.pkl', 'r'))
                left = margins[doc.id]['left']
                right = margins[doc.id]['right']
                strips.trim(left, right)
            print 'Elapsed time: %.2fs' % (clock() - t0)
            sys.stdout.flush()
            
            print 'Building algorithm object for %s ... ' % doc.id
            sys.stdout.flush()  
            t0 = clock()
            seed = (hash(doc.id) + config.seed) % 4294967295
            alg = Proposed(strips, seed=seed, verbose=True, trailing=8)
            print 'Elapsed time: %.2fs' % (clock() - t0)
            sys.stdout.flush() 
            
            for solver, solve in solvers.items():
        
                # Skip solver
                if index + nruns * ns <= index_backup:
                    index += nruns * ns
                    continue
                
                for run in range(1, nruns + 1):
                    
                    # Skip run
                    if index + ns <= index_backup:
                        index += ns
                        continue
                    
                    done = 100 * float(index + ns) / total
                    print '%s[%.2f%%] dataset=%s doc=%s solver=%s run=%d' % \
                        (4 * ' ', done, dataset.id, doc.id, solver, run)
                    sys.stdout.flush()
                    
                    print '%sMatrix ... ' % (4 * ' ')
                    sys.stdout.flush() 
                    t0 = clock()
                    matrix = alg(**params).matrix
                    print '%sElapsed time: %.2fs' % (4 * ' ', clock() - t0)
                    sys.stdout.flush() 
                    
                    print '%sSolution ... ' % (4 * ' '),
                    sys.stdout.flush()
                    t0 = clock()
                    for s, mat in enumerate(matrix, 1):
                        nwords = float('nan')
                        acc = float('nan')
                        sol = None
                        if check_matrix(mat):
                            sol, cost = solve(mat)
                            if sol is not None:
                                acc = accuracy(sol)
                                nwords = number_of_words(
                                    strips.image(sol), dataset.language
                                )
                        sol = ' '.join(str(v) for v in sol) if sol is not None else ''
                    
                        # Storing
                        df_proposed.loc[index] = [solver, dset_main_id, dset_shredding,
                           doc.id, run, s, nwords, acc, cost, sol]
                        index += 1
                    print '%sElapsed time: %.2fs' % (4 * ' ', clock() - t0)
                    sys.stdout.flush() 
                
                    # Dumping
                    df_proposed.to_csv(
                        df_filename, sep='\t', encoding='utf-8', index=False,
                        mode='a', header=False
                    )
                    df_proposed.drop(df_proposed.index, inplace=True) # clear
            
            gc.collect()
            print '%d items in garbage' % (len(gc.garbage))
            sys.stdout.flush()
except Exception as e:
    from myemail import send_mail
    send_mail(
        config.email, config.pwd, config.email,
        subject='Status testing_proposed.py', txt=str(e)
    )

