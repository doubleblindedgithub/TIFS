import sys
sys.path.append('../')

import os
import cv2
import shutil
from shredding import shred

src_path = '../dataset/training/full_images'
dst_path = '../dataset/training/artificial'

if os.path.exists(dst_path):
    shutil.rmtree(dst_path)
os.makedirs(dst_path)

docs = sorted(os.listdir(src_path))
for i, doc in enumerate(docs, 1):
    new_doc = 'D%03d' % i
    os.makedirs(os.path.join(dst_path, new_doc, 'strips'))
    image = cv2.imread(os.path.join(src_path, doc))
    strips = shred(image, apply_noise=True, sigma=200)
    for i, strip in enumerate(strips, 1):
        filename = os.path.join(
            dst_path, new_doc, 'strips', '%s%02d.jpg' % (new_doc, i)
        )
        cv2.imwrite(filename, strip)
