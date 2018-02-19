'''
USAGE python testing_others.py
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
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path
from docrec.reconstruction.jigsaw.jigsaw import solve_from_matrix
from docrec.reconstruction.compatibility.andalo import Andalo
from docrec.reconstruction.compatibility.balme import Balme
from docrec.reconstruction.compatibility.marques import Marques
from docrec.reconstruction.compatibility.morandell import Morandell
from docrec.reconstruction.compatibility.sleit import Sleit
from docrec.validation.config.experiments import ExperimentsConfig

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

# Algorithms
algorithms = config.algorithms_config.algorithms
del algorithms['proposed']
nalgorithms = len(algorithms)

#==============================================================================
# Other methods
#==============================================================================

# Resulting dataframe
index = index_backup = 0
df_others = pd.DataFrame(
    columns=('method', 'solver', 'dataset', 'shredding', 'document', 'disp',
             'accuracy', 'cost', 'solution')
)
df_filename = config.path_cache('testing_others.csv')
if os.path.exists(df_filename):
    index_backup = len(pd.read_csv(df_filename, sep='\t'))
else:
    df_others.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)

ndisp = config.ndisp
total = nalgorithms * nsolvers * ndocs_total * ndisp

try:
    for alg_id, algorithm in algorithms.items():

        # Skip algorithm
        if index + ndocs_total * nsolvers * ndisp <= index_backup:
            index += ndocs_total * nsolvers * ndisp
            continue

        for dataset in datasets.values():
        
            # Skip dataset
            ndocs = len(dataset.docs)
            if index + ndocs * nsolvers * ndisp <= index_backup:
                index += ndocs * nsolvers * ndisp
                continue
            
            dset_main_id, dset_shredding = dataset.id.split('-')

            for doc in dataset.docs:
                
                # Skip document
                if index + nsolvers * ndisp <= index_backup:
                    index += nsolvers * ndisp
                    continue
            
                print 'Loadind strips of %s ... ' % doc.id,
                sys.stdout.flush()  
                t0 = clock()
                strips = Strips(path=doc.path, filter_blanks=False)
                if dataset.id != 'D2-mec':
                    margins = pickle.load(open('margins_%s.pkl' % dataset.id, 'r'))
                    left = margins[doc.id]['left']
                    right = margins[doc.id]['right']
                    strips.trim(left, right)
                print 'Elapsed time: %.2fs' % (clock() - t0)
                sys.stdout.flush()
            
                print 'Building algorithm object for %s ... ' % doc.id
                sys.stdout.flush()
                t0 = clock()
                alg = eval('%s(strips)' % algorithm.classname)
                print 'Elapsed time: %.2fs' % (clock() - t0)
                sys.stdout.flush() 
            
                for solver, solve in solvers.items():
        
                    # Skip solver
                    if index + ndisp <= index_backup:
                        index += ndisp
                        continue
                    
                    for disp in range(ndisp):
                        done = 100 * float(index + 1) / total
                        print '%s[%.2f%%] dataset=%s doc=%s solver=%s disp=%d' % \
                        (4 * ' ', done, dataset.id, doc.id, solver, disp)
                        sys.stdout.flush()
                    
                        print '%sMatrix ... ' % (4 * ' '),
                        sys.stdout.flush() 
                        t0 = clock()
                        matrix = alg(d=disp, **algorithm.params).matrix
                        print '%sElapsed time: %.2fs' % (4 * ' ', clock() - t0)
                        sys.stdout.flush() 
                        
                        print '%sSolution ... ' % (4 * ' '),
                        sys.stdout.flush()
                        t0 = clock()
                        acc = float('nan')
                        sol = None
                        if check_matrix(matrix):
                            sol, cost = solve(matrix)
                            if sol is not None:
                                acc = accuracy(sol)
                        sol = ' '.join(str(v) for v in sol) if sol is not None else ''
                    
                        # Storing
                        df_others.loc[index] = [alg_id, solver, dset_main_id,
                            dset_shredding, doc.id, disp, acc, cost, sol]
                        index += 1
                        print '%sElapsed time: %.2fs' % (4 * ' ', clock() - t0)
                        sys.stdout.flush() 
                
                    # Dumping
                    df_others.to_csv(
                        df_filename, sep='\t', encoding='utf-8', index=False,
                        mode='a', header=False
                    )
                    df_others.drop(df_others.index, inplace=True) # clear
            
            gc.collect()
            print '%d items in garbage' % (len(gc.garbage))
            sys.stdout.flush()
except Exception as e:
    from myemail import send_mail
    send_mail(
        config.email, config.pwd, config.email,
        subject='Status testing_others.py', txt=str(e)
    )

