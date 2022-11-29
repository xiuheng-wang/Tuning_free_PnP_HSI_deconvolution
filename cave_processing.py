# coding: utf-8
# Script for processing CAVE dataset downloaded from https://www1.cs.columbia.edu/CAVE/databases/multispectral/
#
# Reference: 
# Tuning-free Plug-and-Play Hyperspectral Image Deconvolution with Deep Priors
# Xiuheng Wang, Jie Chen, Cédric Richard
#
# 2019/10
# Implemented by
# Xiuheng Wang
# xiuheng.wang@oca.eu


import numpy as np
import scipy.io as io
import tifffile as tiff
import os
import random
import glob
import cv2 
import scipy.io as scio

raw = './data/complete_ms_data/'
train_raw_dir = './data/train/'
test_raw_dir = './data/test/'

num_train = 20 # num of raw train images

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

if __name__ == '__main__':

    if not os.path.exists(train_raw_dir):
        makedir(train_raw_dir)
    if not os.path.exists(test_raw_dir):
        makedir(test_raw_dir)

    print('begin processing...')
    i = 0
    roots = dict()
    for root, dirs, files in os.walk(raw):
        roots[i] = root
        i += 1
    del roots[0]
    for i in range(0, 64):
        if i%2:
            del roots[i]
    roots = list(roots.values()) # dict --> list
    # random.Random(487).shuffle(roots) 
    img = np.zeros([31, 512, 512]) # 31
    for i, root in enumerate(sorted(roots)):
        # print(i, root)
        png_list = sorted(glob.glob(root+'/*.png'))
        # print(png_list)
        for j, path in enumerate(png_list):
            img[j, :, :] = cv2.imread(path, 0) 
        img = img.astype(np.uint8)

        if i < num_train:
            tiff.imsave(train_raw_dir + str(i) + '.tif', img)
        else:
            tiff.imsave(test_raw_dir + str(i - 20) + '.tif', img)
            scio.savemat(test_raw_dir + str(i - 20) + '.mat', {'img':img})
    print('Done')
    