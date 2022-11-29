# coding: utf-8
# Script for performing hyperspectral image deconvolution for the CAVE dataset
#
# Reference: 
# Tuning-free Plug-and-Play Hyperspectral Image Deconvolution with Deep Priors
# Xiuheng Wang, Jie Chen, Cédric Richard
#
# 2019/10
# Implemented by
# Xiuheng Wang
# xiuheng.wang@oca.eu

from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DnCNN import Net_0
import numpy as np
from save_image import save_image
import time
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
from functions import makedir, psf2otf, center_crop, search_gss, measure
from save_image import save_image, rmse
import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import scipy.io as scio
import argparse

# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='CAVE database deblurring:')
# model storage path
parser.add_argument('--model_path', type=str, default='./models/', help='Set model storage path')
# blurring images path
parser.add_argument('--inputs_path', type=str, default='/blurred_1/', help='Set results storage path')
# results storage path
parser.add_argument('--results_path', type=str, default='/deblurred_1/', help='Set results storage path')
# results storage path
parser.add_argument('--kernel_name', type=str, default='kernel_1', help='Choose the kernel name')

args = parser.parse_args()

# select blur kernel
kernel = scio.loadmat('./data/kernels/' + args.kernel_name + '.mat')['kernel']

raw_image_dir = './data/test/'
blurred_image_dir = './data/blurred' + args.inputs_path
deblurred_image_dir = './data/deblurred' + args.results_path
num_images = 12
model_path = args.model_path + 'hsidb_epoch500.pkl'

# hyperparameters
rho_1 = 0
rho_2 = 1
ell = 1e-3
zeta = 2e-4
Iteration = 30


# Initialize denoise nerual network: 1 --> GPU mode, 0 --> Cpu mode
# It is strongly recommended to use GPU mode as its speed is extremely faster than CPU mode.
# If your GPU memory is limited, we recommend to crop the input of the 3DDnCNN into serveral pathes
# and integrate them after denoising.
mode = 1 
if mode:
##### GPU mode #####
    device="cuda:0"
    model = Net_0(8, 32)
    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    model = nn.DataParallel(model)
    model.load_state_dict(model_param)
else:
##### CPU mode #####
    device="cpu"
    model = Net_0(8, 32)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in pretrain.items():
        if k=='state_dict':
            state_dict=OrderedDict()
            for keys in v:
                name = keys[7:]# remove `module.`
                state_dict[name] = v[keys]
                new_state_dict[k]=state_dict
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict['state_dict'])

# convert model to device
model=model.to(device=device, dtype=torch.float)
model.eval()

if __name__ == '__main__':

    if not os.path.exists(deblurred_image_dir):
        makedir(deblurred_image_dir)

    # for j in range(8):
    plt.figure(0)
    RMSE = np.zeros([num_images, Iteration + 1])
    SAM = np.zeros([num_images, Iteration + 1])
    RHO = np.zeros([num_images, Iteration])
    MRX = np.zeros([num_images, Iteration + 1])

    for i in range(num_images):
        img = scio.loadmat(raw_image_dir + str(i) + '.mat')['img']
        # img = tiff.imread(raw_image_dir + str(i) + '.tif')
        dim = np.shape(img)

        # getting blurred images
        img_blurred = scio.loadmat(blurred_image_dir + str(i) + '.mat')['img_blurred']
        img_blurred = np.clip(img_blurred, 0, 1)

        img = img.astype(np.float32) / 255 # Normalized
        print('deblurring No ' + str(i) + ' image...')
        print('before deblurring: ', 'RMSE: ', rmse(img, img_blurred))

        # deblurring these images
        band = dim[0]
        height = dim[1]
        width = dim[2]

        H = psf2otf(kernel, (height, width))
        H = np.tile(np.expand_dims(H, 0), [band,1,1])

        HTH = abs(H) ** 2
        HT = H.conjugate()
        HTY = HT * np.fft.fftn(img_blurred)

        RMSE[i, 0] = rmse(img, img_blurred)

        # ADMM
        # Initialize variables
        x = img_blurred
        z = x
        u = np.zeros(dim).astype(np.float32)
        flag = 0
        for iter in range(Iteration):
            print("Iter " + str(iter+1) + ":")

            z_tilde = x + u
            z_tilde = np.clip(z_tilde, 0, 1)
            z_tilde = torch.from_numpy(z_tilde).unsqueeze(0).unsqueeze(0)
            z_tilde = z_tilde.to(device=device, dtype=torch.float)
            
            z = model(z_tilde) # Please crop z_tilde into serveral patches if the GPU memory is limited.
            z = np.squeeze(z.to('cpu').detach().numpy())
            print('RMSE: ', rmse(z, img))

            x_tilde = z - u
            x_tilde = np.clip(x_tilde, 0, 1)
            x, rho = search_gss(img_blurred, HTY, x_tilde, HTH, H, rho_1, rho_2, ell)
            RHO[i, iter] = rho
            print('rho:', rho, 'RMSE: ', rmse(x, img))

            r_x = (img_blurred - np.real(np.fft.ifftn(H * np.fft.fftn(x))))
            MRX[i, iter+1] = measure(r_x)
            print("measure: ", MRX[i, iter+1])

            u = u + x - z
            
            RMSE[i, iter+1] = rmse(x, img)

            if (np.abs(MRX[0, iter+1] - MRX[0, iter]) / MRX[0, iter+1] < zeta and flag == 0) or (iter+1 == Iteration and flag == 0 ):
                MRX[i, 0] = iter+1
                save_image(x, img, deblurred_image_dir, i)
                print('--------------finish deblurring-------------: ', 'RMSE: ', rmse(x, img))
                flag = 1
                break
        print('Done')

    scio.savemat(deblurred_image_dir + 'RMSE.mat', {'RMSE':RMSE})
    plt.figure()
    for i in range(num_images):
        plt.plot(RMSE[i])
    plt.savefig(deblurred_image_dir + 'RMSE.png')
    plt.close()

    scio.savemat(deblurred_image_dir + 'RHO.mat', {'RHO':RHO})
    plt.figure()
    for i in range(num_images):
        plt.plot(RHO[i])
    plt.savefig(deblurred_image_dir + 'RHO.png')
    plt.close()

    scio.savemat(deblurred_image_dir + 'MRX.mat', {'MRX':MRX})
    plt.figure()
    for i in range(num_images):
        plt.plot(MRX[i, 1:])
    plt.savefig(deblurred_image_dir + 'MRX.png')
    plt.close()


