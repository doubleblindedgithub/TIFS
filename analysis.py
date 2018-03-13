'''
USAGE python analysis.py
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pickle
from docrec.validation.config.experiments import ExperimentsConfig

# Global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# (1) Accuracy analysis

# Dataset 1 (category info)
categories = pickle.load(open('categories_D1-mec.pkl', 'r'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()
print '#documents in D1 (per categories): to = %d, lg = %d, fg = %d' % \
    (len(categories['to']), len(categories['lg']), len(categories['fg']))

# Data to be analyzed
df_filename = config.path_cache('testing_proposed.csv')
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

df_prop = df[df.solver == 'proposed']
keys = ['dataset', 'shredding', 'document', 'run']

df_prop_glob = df_prop.copy()
df_best_glob = df_prop_glob.loc[df_prop_glob.groupby(keys)['nwords'].idxmax()].reset_index()

df_best_d1 = df_best_glob[df_best_glob.dataset == 'D1'].copy()
df_best_d2 = df_best_glob[df_best_glob.dataset == 'D2'].copy()
#df_best_glob['dataset'] = 'D1 + D2'

#data = pd.concat([df_best_glob, df_best_d1, df_best_d2])

keys = ['document']
df_mean_d1_mec = df_best_d1[df_best_d1.shredding == 'mechanical'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')
df_mean_d1_art = df_best_d1[df_best_d1.shredding == 'artificial'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')
df_mean_d2_mec = df_best_d2[df_best_d2.shredding == 'mechanical'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')
df_mean_d2_art = df_best_d2[df_best_d2.shredding == 'artificial'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')

df_mean_d1_mec['category'] = df_mean_d1_mec.document.map(doc_category_map)
df_mean_d1_art['category'] = df_mean_d1_art.document.map(doc_category_map)

df_mean_d1_mec_to = df_mean_d1_mec[df_mean_d1_mec.category == 'TO']
df_mean_d1_art_to = df_mean_d1_art[df_mean_d1_art.category == 'TO']
df_mean_d1_mec_lg = df_mean_d1_mec[df_mean_d1_mec.category == 'LG']
df_mean_d1_art_lg = df_mean_d1_art[df_mean_d1_art.category == 'LG']
df_mean_d1_mec_fg = df_mean_d1_mec[df_mean_d1_mec.category == 'FG']
df_mean_d1_art_fg = df_mean_d1_art[df_mean_d1_art.category == 'FG']

print 'Accuracy for D1 (to): %.2f' % df_mean_d1_mec_to.accuracy.mean()
print 'Accuracy decay for D1: %.2f' % (df_mean_d1_art.accuracy.mean() - df_mean_d1_mec.accuracy.mean())
print 'Accuracy decay for D1 (to): %.2f' % (df_mean_d1_art_to.accuracy.mean() - df_mean_d1_mec_to.accuracy.mean())
print 'Accuracy decay for D1 (lg): %.2f' % (df_mean_d1_art_lg.accuracy.mean() - df_mean_d1_mec_lg.accuracy.mean())
print 'Accuracy decay for D1 (fg): %.2f' % (df_mean_d1_art_fg.accuracy.mean() - df_mean_d1_mec_fg.accuracy.mean())

df_mean_d1_mec_to_la = df_mean_d1_mec_to.copy().sort_values(['accuracy'])
df_mean_d1_mec_lg_la = df_mean_d1_mec_lg.copy().sort_values(['accuracy'])
df_mean_d1_mec_fg_la = df_mean_d1_mec_fg.copy().sort_values(['accuracy'])

print 'Sorted accuracy'
print df_mean_d1_mec_to_la
print df_mean_d1_mec_lg_la
print df_mean_d1_mec_fg_la

print 'Accuracy for D2: %.2f' % df_mean_d2_mec.accuracy.mean()
print 'Accuracy decay for D2: %.2f' % (df_mean_d2_art.accuracy.mean() - df_mean_d2_mec.accuracy.mean())

keys = ['dataset', 'shredding', 'document', 'run']

print 'OCR-aided filtering effectiveness only for text'
d1 = df_prop_glob[df_prop_glob.dataset == 'D1']
d1['category'] = d1.document.map(doc_category_map)
d1 = d1[d1.category == 'TO'].drop(['category'], axis=1)
d2 = df_prop_glob[df_prop_glob.dataset == 'D2']
df_ocr = pd.concat([d1, d2])

acc_words = df_ocr.loc[df_ocr.groupby(keys)['nwords'].idxmax()].reset_index(drop=True).accuracy
acc = df_ocr.loc[df_ocr.groupby(keys)['accuracy'].idxmax()].reset_index(drop=True).accuracy

print acc_words
print acc
print 'AQUI'

#df_best_acc_glob = df_prop_glob.loc[df_prop_glob.groupby(keys)['accuracy'].idxmax()].reset_index()
p = float((acc == acc_words).sum()) / len(acc)

#p = float((df_best_acc_glob.accuracy == df_best_glob.accuracy).sum()) / len(df_best_acc_glob)

#df_best_acc_glob['category'] = df_best_acc_glob.document.map(doc_category_map)
#print df_best_acc_glob

print 'Choice of best solution: %.2f%%' % (100 * p)

#print '#words for well-chosen solutions' 
#print df_best_glob[df_best_acc_glob.accuracy == df_best_glob.accuracy].nwords.mean()

#print '#words for badly-chosen solutions'
#print df_best_glob[df_best_acc_glob.accuracy != df_best_glob.accuracy].nwords.mean()

#df_diff = (df_mean_d1_art.accuracy - df_mean_d1_mec.accuracy).reset_index(name='accuracy')
#df_diff['document'] = df_mean_d1_art.document
#df_diff['category'] = df_diff.document.map(doc_category_map)

#df_bad = df_diff[df_diff.accuracy > 0.2]
#print '#documents for which accuracy fell more than 20%%: %d / 60 (%.2f%%)' % (len(df_bad), 100 * len(df_bad) / 60.0)
      
#df_bad_to = df_bad[df_bad.category == 'TO']
#df_bad_lg = df_bad[df_bad.category == 'LG']
#df_bad_fg = df_bad[df_bad.category == 'FG']
#n_to = len(categories['to'])
#n_lg = len(categories['lg'])
#n_fg = len(categories['fg'])
#print('TO: %d / %d' % (len(df_bad_to), n_to))
#print('LG: %d / %d' % (len(df_bad_lg), n_lg))
#print('FG: %d / %d' % (len(df_bad_fg), n_fg))

df_filename = config.path_cache('testing_others.csv')
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df.method.replace({'andalo2': 'andalo1'}, inplace=True)
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

df_solver = df[df.solver == 'proposed']
keys = ['method', 'dataset', 'shredding', 'document']

df_others_glob = df_solver.copy()
df_others_best_glob = df_others_glob.loc[df_others_glob.groupby(keys)['accuracy'].idxmax()].reset_index()
print df_others_glob
#df_best_glob['dataset'] = 'D1 + D2'
#df_others_best_glob['dataset'] = 'D1 + D2'
df_best_glob['method'] = 'proposed'
#df_others_best_glob.method.replace({'andalo2': 'andalo1'}, inplace=True)

data = pd.concat([df_best_glob, df_others_best_glob])
keys = ['method', 'shredding']
print 'Global results'
print data.groupby(keys)['accuracy'].mean()

keys = ['method', 'dataset', 'shredding']
print 'Results by dataset'
print data.groupby(keys)['accuracy'].mean()

keys = ['method', 'shredding', 'category']
data = data[data.dataset == 'D1']
data['category'] = data.document.map(doc_category_map)
print 'Results by category (D1)'
print data.groupby(keys)['accuracy'].mean()

# (2) Time analysis
#df_filename = config.path_cache('timing.csv')
#df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
#df = df[~ np.isnan(df.accuracy)]
#df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})




