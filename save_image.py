import numpy as np
import math
import tifffile as tiff
import os
import scipy.io as scio

def save_image(output, label, out_results_path, filename):
    if not os.path.exists(out_results_path):
        os.makedirs(out_results_path)
    image = output
    image = np.clip(image, 0, 1) 
    image = image * 255
    tiff.imsave(out_results_path + str(filename) + '.tif', image.astype(np.uint8))
    
    output = output.transpose([1, 2, 0])
    label = label.transpose([1, 2, 0])
    scio.savemat(out_results_path + str(filename) + '.mat', {'sr':output, 'gt':label})

def save_image_real(output, out_results_path, filename):
    if not os.path.exists(out_results_path):
        os.makedirs(out_results_path)
    image = output
    image = np.clip(image, 0, 1) 
    image = image * 255
    tiff.imsave(out_results_path + str(filename) + '.tif', image.astype(np.uint8))
    scio.savemat(out_results_path + str(filename) + '.mat', {'sr':image.astype(np.uint8)})

def rmse(img1, img2):
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    img1 = (img1*255).astype(np.uint8)
    img2 = (img2*255).astype(np.uint8)
    aux = np.mean((img1 - img2) ** 2)
    return np.sqrt(aux)

