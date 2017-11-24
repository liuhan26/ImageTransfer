#coding:utf-8
import os
from random import shuffle
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from PIL import Image
from skimage import io
import scipy.misc
def load_data(dataset):
   class1_files = []
   with open('rgbd/color.txt') as f:
       for line in f.readlines():
           class1_files.append(line[0:-3])

   class2_files = []
   with open('rgbd/depth.txt') as f:
       for line in f.readlines():
           class2_files.append(line[0:-3])

   return class1_files, class2_files
