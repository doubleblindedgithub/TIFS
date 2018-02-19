import numpy as np
import sys
import cv2
import pandas as pd
from docrec.strips.strips import Strips
from docrec.validation.config.experiments import ExperimentsConfig

config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

df_filename = config.path_cache('testing_proposed.csv')
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})
df_prop = df[df.solver == 'proposed'].reset_index()

keys = ['run']
for doc in ['D002', 'D009', 'D057']:
    df_doc = df_prop[(df_prop.dataset == 'D1') & (df_prop.document == doc) & (df_prop.shredding == 'mechanical')]
    mean = df_doc.loc[df_doc.groupby(keys)['nwords'].idxmax()].accuracy.mean()
    acc = df_doc.loc[(df_doc.accuracy - mean).abs().idxmin()].accuracy
    print 'Doc=%s Acc=%.2f' % (doc, acc)
    sol = df_doc.loc[(df_doc.accuracy - mean).abs().idxmin()].solution
    sol = map(int, sol.split())
    strips = Strips(path='dataset/D1/mechanical/%s' % doc)
    img = strips.image(order=sol)
    cv2.imwrite(
        'results/%s_rec.jpg' % doc,
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )
