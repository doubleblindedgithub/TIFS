'''
USAGE python graphs.py
'''
from docrec.validation.config.experiments import ExperimentsConfig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pickle
import seaborn as sns
sns.set(
    context='paper', style='darkgrid', palette='deep', font_scale=0.8
)

# Dataset 1 (category info)
categories = pickle.load(open('categories_D1-mec.pkl', 'r'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()

# Global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# Data to be analyzed
df_filename = config.path_cache('testing_proposed.csv')
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

# 1) Proposed system -------------------------------------------------------------
df_solver = df[df.solver == 'proposed']
keys = ['dataset', 'shredding', 'document', 'run']

df_prop_glob = df_solver.copy()
df_prop_best_glob = df_prop_glob.loc[df_prop_glob.groupby(keys)['nwords'].idxmax()].reset_index()

df_prop_best_d1 = df_prop_best_glob[df_prop_best_glob.dataset == 'D1'].copy()
df_prop_best_d2 = df_prop_best_glob[df_prop_best_glob.dataset == 'D2'].copy()
df_prop_best_glob['dataset'] = 'D1 + D2'

data_prop = pd.concat([df_prop_best_glob, df_prop_best_d1, df_prop_best_d2])
data_prop.drop(['s'], axis=1, inplace=True)
data_prop['method'] = 'proposed'

fp = sns.factorplot(
    x='dataset', y='accuracy', data=data_prop,
    hue='shredding', kind='box', size=1.85, aspect=1.5,
    margin_titles=True, fliersize=2, width=0.65, linewidth=0.5,
    legend=True
)

#df_acc = df_solver_d2.groupby(keys)['accuracy'].max().reset_index(name='accuracy')
plt.savefig(config.path_results('g1.pgf'), bbox_inches='tight')

df_prop_best_d1['category'] = df_prop_best_d1.document.map(doc_category_map)
fp = sns.factorplot(
    x='category', y='accuracy', data=df_prop_best_d1,
    hue='shredding', kind='box', size=1.85, aspect=1.5,
    order=['TO', 'LG', 'FG'], margin_titles=True, fliersize=2,
    width=0.65, linewidth=0.5, legend=True, legend_out=True
)

plt.savefig(config.path_results('g2.pgf'), bbox_inches='tight')

# Data to be analyzed
df_filename = config.path_cache('testing_others.csv')
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df.method.replace({'andalo2': 'andalo1'}, inplace=True)
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

df_solver = df[df.solver == 'proposed']
keys = ['method', 'dataset', 'shredding', 'document']

df_others_glob = df_solver.copy()
df_others_best_glob = df_others_glob.loc[df_others_glob.groupby(keys)['accuracy'].idxmax()].reset_index()

df_others_best_d1 = df_others_best_glob[df_others_best_glob.dataset == 'D1'].copy()
df_others_best_d2 = df_others_best_glob[df_others_best_glob.dataset == 'D2'].copy()
df_others_best_glob['dataset'] = 'D1 + D2'

data_others = pd.concat([df_others_best_glob, df_others_best_d1, df_others_best_d2])
data_others.drop(['disp'], axis=1, inplace=True)
data = pd.concat([data_prop, data_others])

legends = [algorithm.legend
           for algorithm in config.algorithms_config.algorithms_list
           if algorithm.id in data.method.unique()
          ]
legends_map = {algorithm.id: algorithm.legend for algorithm in config.algorithms_config.algorithms_list}
data.method.replace(legends_map, inplace=True)

fp = sns.factorplot(
    x='dataset', y='accuracy', data=data[data.shredding == 'mechanical'],
    hue='method', hue_order=legends, kind='box', size=2.0, aspect=2.5,
    margin_titles=True, fliersize=2, width=0.65, linewidth=0.5,
    legend=True
)
legend = fp._legend
legend.set_bbox_to_anchor((0.95, 0.5))
plt.savefig(config.path_results('g3.pgf'), bbox_inches='tight')
#legend.set_title('Method')
#for t, l in zip(legend.texts, legends):
#    t.set_text(r'%s'% l)

data_d1 = data[data.dataset == 'D1'].copy()
data_d1['category'] = data_d1.document.map(doc_category_map)
data_d1.method.replace(legends_map, inplace=True)
fp = sns.factorplot(
    x='category', y='accuracy', order=['TO', 'LG', 'FG'], hue_order=legends,
    data=data_d1[data_d1.shredding == 'mechanical'],
    hue='method', kind='box', size=2.0, aspect=2.5,
    margin_titles=True, fliersize=2, width=0.65, linewidth=0.5,
    legend=True
)
legend = fp._legend
legend.set_bbox_to_anchor((0.94, 0.5))
plt.savefig(config.path_results('g4.pgf'), bbox_inches='tight')

fp = sns.factorplot(
    x='method', y='accuracy', order=legends, data=data[data.dataset == 'D1 + D2'],
    hue='shredding', kind='box', size=2.0, aspect=2.5,
    margin_titles=True, fliersize=2, width=0.5, linewidth=0.5,
    legend=True
)
plt.savefig(config.path_results('g5.pgf'), bbox_inches='tight')

#plt.show()



#
#
#df_prop_d1 = df_prop[df_prop.dataset.isin(['D1-mec', 'D1-art'])]
#df_best = df_prop_d1.loc[df_prop_d1.groupby(keys)['nwords'].idxmax()].reset_index()
#df_best['category'] = df_best.document.map(doc_category_map)
#order = ['TO', 'LG', 'FG']
#bp = sns.boxplot(
#    x='category', y='accuracy', hue='dataset', order=order, data=df_best,
#    width=0.4, fliersize=2, linewidth=1, ax=ax2
#)
#plt.legend(loc='lower left')
#
#df_prop_d2 = df_prop[df_prop.dataset.isin(['D2-mec', 'D2-art'])]
#df_best = df_prop_d2.loc[df_prop_d2.groupby(keys)['nwords'].idxmax()].reset_index()
#df_best['category'] = ''
#
#bp = sns.boxplot(
#    x='category', y='accuracy', hue='dataset', data=df_best,
#    width=0.2, fliersize=2, linewidth=1, ax=ax3
#)
#plt.legend(loc='lower left')
#
##df_acc = df_prop_d1.groupby(keys)['accuracy'].max().reset_index(name='accuracy')
#
##ax1.legend_.remove()
##ax1.legend()
#ax1.set_xlabel('Global')
#ax2.set_xlabel('D1')
#ax2.set_ylabel('')
##ax2.set_yticklabels([''])
#ax3.set_xlabel('D2')
#ax3.set_ylabel('')
#ax3.set_xticklabels(['s'])
#ax3.set_yticklabels([])
#
#plt.savefig(config.path_results('g1.pgf'), bbox_inches='tight')
#plt.show()
#

#df_prop_d2 = df_prop[df_prop.dataset.isin(['D2-mec', 'D2-art'])]
#df_best = df_prop_d2.loc[df_prop_d2.groupby(keys)['nwords'].idxmax()].reset_index()
#df_best['category'] = ''
#
#bp = sns.boxplot(
#    x='category', y='accuracy', hue='dataset', data=df_best,
#    width=0.2, fliersize=2, linewidth=1, ax=ax3
#)
#plt.legend(loc='lower left')



#x = (df_best.accuracy == df_acc.accuracy)


# Graph 2: Proposed, Dataset 2 ------------------------------------------------
#fig = plt.figure(figsize=(1.5, 2), dpi=300)
#df_prop_d2 = df_prop[df_prop.dataset.isin(['D2-mec', 'D2-art'])]
#
#keys = ['dataset', 'document', 'run']
#df_best = df_prop_d2.loc[df_prop_d2.groupby(keys)['nwords'].idxmax()].reset_index()
#df_best['category'] = ''
#
#bp = sns.boxplot(
#    x='category', y='accuracy', hue='dataset', data=df_best,
#    width=0.2, fliersize=2, linewidth=1
#)
#
#bp.axes.set_xlabel('')
#bp.axes.set_xticklabels([''])
#plt.legend(loc='lower left')
#df_acc = df_prop_d2.groupby(keys)['accuracy'].max().reset_index(name='accuracy')
#plt.savefig(config.path_results('g2.pgf'), bbox_inches='tight')
#plt.show()
#x = (df_best.accuracy == df_acc.accuracy)


# Graph 2: Proposed, Dataset 2 ------------------------------------------------
#        hue_order = [algorithm.id for algorithm in algorithms]
#        fp = sns.factorplot(
#            x='Category', y='Accuracy', data=df,
#            hue='Method', row=row, col='Dataset', kind='box', order=order,
#            hue_order=hue_order, palette='Set2', size=1.5, aspect=1.5,
#            margin_titles=True, fliersize=2, width=0.65, linewidth=0.5,
#            legend=False
#        )


#        self.data = self._format_data(df, strategy)
#
#        if experiment == 1:
#            # Add a new column: fixing option in function of solver
#            df['fixing_option'] = df.solver.map(
#                {'prop': 'None', 'prop_s': 'S', 'prop_se': 'SE'}
#            )
#

#for strategy in ['local', 'global', 'maxdoc']:
#    graph = Graph(config, strategy=strategy)
#    for experiment, dataset in product([1, 2, 3], [1, 2]):
#        filename = config.path_results(
#            'graph_%d-%d-%s.pgf' % (experiment, dataset, strategy))
#        
#        print 'Plotting %s:' % filename,
#        graph.plot(2, experiment, dataset, True)
#        graph.write(filename)
#        print 'done!'
        
        
