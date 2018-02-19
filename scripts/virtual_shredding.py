import sys
sys.path.append('../')

import os
import numpy as np
from numpy.random import randn, randint, standard_normal
import cv2
import shutil
from shredding import shred


assert len(sys.argv) > 1

option = sys.argv[1]
if option == 'marques_2013':
    mec_path = '../dataset/marques_2013/mechanical'
    art_path = '../dataset/marques_2013/artificial'
    if os.path.exists(art_path):
        shutil.rmtree(art_path)
    os.makedirs(art_path)
    docs = ['D001', 'D002'] + ['D%03d' % i for i in range(4, 62)]
    for doc in docs:
        os.makedirs(os.path.join(art_path, doc, 'strips'))
        image = cv2.imread(os.path.join(mec_path, doc, '%s.jpg' % doc))
        strips = shred(image)
        for i, strip in enumerate(strips, 1):
            filename = os.path.join(
                art_path, doc, 'strips', '%s%02d.jpg' % (doc, i)
            )
            cv2.imwrite(filename, strip)
            print filename

elif option == 'andalo_2017':
    int_path = '../dataset/andalo_2017/integral'
    art_path = '../dataset/andalo_2017/artificial'
    if os.path.exists(art_path):
        shutil.rmtree(art_path)
    os.makedirs(art_path)
    docs = ['D%03d' % i for i in range(1, 21)]
    for doc in docs:
        os.makedirs(os.path.join(art_path, doc, 'strips'))
        image = cv2.imread(os.path.join(int_path, '%s.TIF' % doc))
        strips = shred(image)
        for i, strip in enumerate(strips, 1):
            filename = os.path.join(
                art_path, doc, 'strips', '%s%02d.jpg' % (doc, i)
            )
            cv2.imwrite(filename, strip)
            print filename

