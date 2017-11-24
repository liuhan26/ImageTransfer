#coding:utf-8
import os
from random import shuffle
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from PIL import Image
from skimage import io
import scipy.misc
def load_data(dataset, split = "train", percentage = 0.8):
   rootdir1 = "/media/liuhan/xiangziBRL/Lock3DFace/croppedData_LightenedCNN/color/FE"
   if dataset == "lock3dface":
           #data_dir = './data/lock3dface/color/'
           #file_list = tl.files.load_file_list(path=data_dir, regx='\.(jpg)',printable=False)
           #class1_files = []
           i=0
           for lists in os.listdir(rootdir1):
               path = os.path.join(rootdir1,lists)
               for files in os.listdir(path):
                    filepath = os.path.join(path,files)
                    img = io.imread(filepath)
                    img = img.reshape((144,144,1))
                    img = tl.prepro.imresize(img, size=[64, 64], interp='bilinear', mode=None)
                    img = np.repeat(img, 3, axis=2)
                    scipy.misc.imsave('data/dataset/color/test_{}.png'.format(i), img)
                    break
               i=i+1

   rootdir2 = "/media/liuhan/xiangziBRL/Lock3DFace/croppedData_LightenedCNN/depth/FE"
   if dataset == "lock3dface":
           #class2_files = []
           i=0
           for lists in os.listdir(rootdir2):
               path = os.path.join(rootdir2,lists)              
               for files in os.listdir(path):
                    filepath = os.path.join(path,files)
                    img = io.imread(filepath)
                    img = img.reshape((144,144,1))
                    img = tl.prepro.imresize(img, size=[64, 64], interp='bilinear', mode=None)
                    img = np.repeat(img, 3, axis=2)
                    scipy.misc.imsave('data/dataset/depth/test_{}.png'.format(i), img)
                    break
               i=i+1
   file_list = tl.files.load_file_list(path='data/dataset/color', regx='\.(png)', printable=False)
   class1_files = []
   for f in file_list:
            if split == 'train' and 'train' in f:
                class1_files.append("data/dataset/color/" + f)
            if split == 'test' and 'test' in f:
                class1_files.append("data/dataset/color/" + f)

   file_list = tl.files.load_file_list(path='data/dataset/depth', regx='\.(png)', printable=False)
   class2_files = []
   for f in file_list:
            if split == 'train' and 'train' in f:
                class2_files.append("data/dataset/depth/" + f)
            if split == 'test' and 'test' in f:
                class2_files.append("data/dataset/depth/" + f)
   shuffle(class1_files)
   shuffle(class2_files)
   class_flag = {}
   for file_name in class1_files:
            class_flag[file_name] = True

   for file_name in class2_files:
            class_flag[file_name] = False
   return class1_files, class2_files, class_flag
