import numpy as np
import os
from skimage.io import imread
from ..ndarray.utils import first_nonzero, last_nonzero


def join(image1, image2, offset1=0, offset2=0, cval=0):
   ''' Join images horizontally considering offset. '''
   
   h1, w1 = image1.shape
   h2, w2 = image2.shape
   
   h = max(h1 + offset1, h2 + offset2)
   w = w1 + w2
   
   joined = np.empty((h, w), dtype=image1.dtype)
   joined[:] = cval
   joined[offset1 : offset1 + h1, : w1] = image1
   joined[offset2 : offset2 + h2, w1 :] = image2
   
   return joined

#def vshift(img, offset=0, cval=0):
#   return np.pad(img,((offset,0),(0,0)), mode='constant', constant_values=cval)
#
#def hshift(img, offset=0, cval=0):
#   return np.pad(img,((0,0),(offset,0)), mode='constant', constant_values=cval)
#
## Trail zeros
#def trail(img):
#   vert_proj_hist = img.sum(axis=0)
#   hor_proj_hist = img.sum(axis=1) 
#   
#   top = first_nonzero(hor_proj_hist)
#   bottom = last_nonzero(hor_proj_hist)
#   left = first_nonzero(vert_proj_hist)
#   right = last_nonzero(vert_proj_hist)
#   
#   return img[top:bottom+1, left:right+1]
#
#def read_images_from_dir(path):
#   return [imread(os.path.join(path, img_file)) for img_file in os.listdir(path)]
#
#def squeeze(imgs, radius=1, cut=0.5, include_orig=True):
#   imgs_ = list(imgs) if include_orig else [] 
#   if radius >= 1:
#      for img in imgs:
#         w = img.shape[1]
#         m = int(cut*w)
#         for r in xrange(1, radius+1):
#            if (r <= m + 1) and (r <= w - m):
#               imgs_.append(np.hstack((img[:,:m-r+1], img[:,m+r:])))
#   return imgs_
#
#def slide(imgs, vdisp=3, cut=0.5, include_orig=True):
#   imgs_ = list(imgs) if include_orig else []
#   for img in imgs:
#      w = img.shape[1]
#      m = int(cut*w)
#      
#      if (m > 0) and (m < w):
#         for d in xrange(1, vdisp+1):
#            imgs_.append(join(img[:,:m], img[:,m:], d, 0))
#            imgs_.append(join(img[:,:m], img[:,m:], 0, d))
#         
#   return imgs_