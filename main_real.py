# coding: utf-8
# Script for performing hyperspectral image deconvolution for real data
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
from functions import makedir, psf2otf, center_crop, search_gss, measure, add_gaussian_noise, normalizeQuantile
from save_image import save_image_real
import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import scipy.io as scio
import argparse
import copy

# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='HiDe database deblurring:')
# model storage path
parser.add_argument('--model_path', type=str, default='./models/', help='Set model storage path')
# image name
parser.add_argument('--image_name', type=str, default='indoor/alien/', help='Set results storage path')
# blurring images path
parser.add_argument('--inputs_path', type=str, default='blurred', help='Set results storage path')
# results storage path
parser.add_argument('--results_path', type=str, default='deblurred', help='Set results storage path')
# results storage path
parser.add_argument('--kernel_name', type=str, default='kernel', help='Choose the kernel name')

args = parser.parse_args()

# select blur kernel
kernel = scio.loadmat('./data/HiDe/' + args.image_name + args.kernel_name + '.mat')['kernel']
blurred_image_dir = './data/HiDe/' + args.image_name + args.inputs_path
deblurred_image_dir = './data/deblurred_HiDe/' + args.image_name
model_path = args.model_path + 'hsidb_epoch500.pkl'
num_images = 1

# hyperparameters
rho_1 = 0
rho_2 = 10
ell = 1e-3
zeta = 2e-4
Iteration = 30
SNR = 40

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

    # getting blurred images
    img_blurred = scio.loadmat(blurred_image_dir + '.mat')['img']
    img_blurred = img_blurred.transpose([2, 0, 1]).astype(np.double) / 255.0
    img_blurred = np.clip(img_blurred, 0, 1)
    dim = np.shape(img_blurred)

    img_blurred = add_gaussian_noise(img_blurred, SNR)

    # deblurring these images
    band = dim[0]
    height = dim[1]
    width = dim[2]

    H = np.zeros(dim, dtype = complex)
    for i in range(band):
        H[i, :, :] = psf2otf(kernel[:,:,i], (height, width))

    HTH = abs(H) ** 2
    HT = H.conjugate()
    HTY = HT * np.fft.fftn(img_blurred)

    # ADMM
    # Initialize variables
    x = img_blurred
    z = x
    u = np.zeros(dim).astype(np.float32)
    flag = 0
    MRX[0, 0] = 1000000
    for iter in range(Iteration):
        print("Iter " + str(iter+1) + ":")

        z_tilde = x + u
        z_tilde = np.clip(z_tilde, 0, 1)
        z_tilde = torch.from_numpy(z_tilde).unsqueeze(0).unsqueeze(0)
        z_tilde = z_tilde.to(device=device, dtype=torch.float)
        
        z_part = z_tilde[:,:,:,0:390,0:348]
        z_part = z_part.to(device=device, dtype=torch.float)
        z_part = model(z_part)
        z[:,0:390,0:348] = np.squeeze(z_part.to('cpu').detach().numpy())

        z_part = z_tilde[:,:,:,390:780,0:348]
        z_part = z_part.to(device=device, dtype=torch.float)
        z_part = model(z_part)
        z[:,390:780,0:348] = np.squeeze(z_part.to('cpu').detach().numpy())
        
        z_part = z_tilde[:,:,:,0:390,348:696]
        z_part = z_part.to(device=device, dtype=torch.float)
        z_part = model(z_part)
        z[:,0:390,348:696] = np.squeeze(z_part.to('cpu').detach().numpy())

        z_part = z_tilde[:,:,:,390:780,348:696]
        z_part = z_part.to(device=device, dtype=torch.float)
        z_part = model(z_part)
        z[:,390:780,348:696] = np.squeeze(z_part.to('cpu').detach().numpy())

        x_tilde = z - u
        x_tilde = np.clip(x_tilde, 0, 1)

        x_copy = copy.deepcopy(x)
        x, rho = search_gss(img_blurred, HTY, x_tilde, HTH, H, rho_1, rho_2, ell)
        RHO[0, iter] = rho
        print('rho:', rho)

        r_x = (img_blurred - np.real(np.fft.ifftn(H * np.fft.fftn(x))))
        MRX[0, iter+1] = measure(r_x)
        print("measure: ", MRX[0, iter+1])

        u = u + x - z
        if (np.abs(MRX[0, iter+1] - MRX[0, iter]) / MRX[0, iter+1] < zeta and flag == 0) or (iter+1 == Iteration and flag == 0 ):
            MRX[0, 0] = iter
            save_image_real(x_copy, deblurred_image_dir, args.results_path)
            print('--------------finish deblurring-------------')
            flag = 1
            break              
        if ((MRX[0, iter+1] > MRX[0, iter]) and flag == 0) or (iter+1 == Iteration and flag == 0 ):
            MRX[0, 0] = iter
            save_image_real(x_copy, deblurred_image_dir, args.results_path)
            print('--------------finish deblurring-------------')
            flag = 1
            break
    print('Done') 
        
    scio.savemat(deblurred_image_dir + 'RHO_' + args.results_path + '.mat', {'RHO':RHO})
    plt.figure()
    plt.plot(RHO[0])
    plt.savefig(deblurred_image_dir + 'RHO_' + args.results_path + '.png')
    plt.close()

    scio.savemat(deblurred_image_dir + 'MRX_' + args.results_path + '.mat', {'MRX':MRX})
    plt.figure()
    plt.plot(MRX[0, 1:])
    plt.savefig(deblurred_image_dir + 'MRX_' + args.results_path + '.png')
    plt.close()
