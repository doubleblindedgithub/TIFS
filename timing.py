'''
USAGE python cron.py
'''
import sys
import os
import numpy as np
from time import clock
import gc
import cPickle as pickle
import pandas as pd
from docrec.strips.strips import Strips
from docrec.strips.stripschar import StripsChar
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path
from docrec.reconstruction.compatibility.andalo import Andalo
from docrec.reconstruction.compatibility.balme import Balme
from docrec.reconstruction.compatibility.marques import Marques
from docrec.reconstruction.compatibility.morandell import Morandell
from docrec.reconstruction.compatibility.proposed import Proposed
from docrec.reconstruction.compatibility.sleit import Sleit
from docrec.validation.config.experiments import ExperimentsConfig
from docrec.ocr.recognition import number_of_words

# Global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# Solvers
solve = lambda mat: shortest_directed_hamiltonian_path(mat)

# Algorithms
algorithms = config.algorithms_config.algorithms

# Sample document
doc = 'dataset/D1/mechanical/D014'
margins = pickle.load(open('margins_D1-mec.pkl', 'r'))
left = margins['D014']['left']
right = margins['D014']['right']

# Resulting dataframe
index = 0
df = pd.DataFrame(columns=('method', 'run', 'time'))
df_filename = config.path_cache('timing.csv')
df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)

for alg_id, algorithm in algorithms.items():
    if alg_id == 'proposed':
        for run in range(1, config.nruns + 1):
            print 'alg=%s run=%d' % (alg_id, run)
            sys.stdout.flush()
            t0 = clock()
            strips = StripsChar(path=doc, filter_blanks=True)
            seed = (hash('D014') + config.seed) % 4294967295
            alg = Proposed(strips, seed=seed, verbose=False)
            matrix = alg(**algorithm.params).matrix
            gc.collect()
            print '%d items in garbage' % (len(gc.garbage))
            sys.stdout.flush()
            for mat in matrix:
                sol, cost = solve(mat)
		print sol, '\n', cost
                nwords = number_of_words(strips.image(sol), 'pt_BR')
            # Storing
            df.loc[index] = [alg_id, run, clock() - t0]
            index += 1
    else:
	continue
        for run in range(1, config.nruns + 1):
            print 'alg=%s run=%d' % (alg_id, run)
            sys.stdout.flush()
            t0 = clock()
            strips = Strips(path=doc, filter_blanks=False)
            strips.trim(left, right)
            alg = eval('%s(strips)' % algorithm.classname)
            matrix = alg(d=3, **algorithm.params).matrix
            sol, cost = solve(matrix)
            df.loc[index] = [alg_id, run, clock() - t0]
            index += 1

# Dumping
df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)
