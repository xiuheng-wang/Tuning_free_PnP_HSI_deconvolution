import numpy as np
import cv2 
import os
import math
from scipy.io import loadmat
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import scipy.io as scio

def normalizeQuantile(Z, qval):
    bands = Z.shape[0]
    for i in range(bands):
        Z[i, :, :] = Z[i, :, :] / np.quantile(Z[i, :, :].flatten(), qval)
    return Z
    
def add_gaussian_noise(img, snr = 10):
    snr = 10**(snr/10.0)
    xpower = np.sum(img**2)/img.size
    npower = xpower / snr
    noise = np.random.randn(np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]) * np.sqrt(npower)
    img = img + noise
    return img

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def center_crop(img, outsize):
    dim = np.shape(img)[1:3]
    a = int((dim[0]-outsize[0])/2)
    b = int((dim[1]-outsize[1])/2)
    img_cropped = img[:, a:a+outsize[0], b:b+outsize[1]]
    return img_cropped

def measure(img):
    # dim = np.shape(img)
    # img = img[:, dim[1]//2 - 50: dim[1]//2 + 50, dim[2]//2 - 50: dim[2]//2 + 50]
    dim = np.shape(img)
    ac = scipy.signal.correlate(img, img, mode='full', method='fft')
    ac = ac[dim[0]-1:, dim[1]-1:, dim[2]-1:]
    meansure = ac / np.linalg.norm(img)**2
    meansure = np.linalg.norm(meansure)**2
    return meansure

def search_gss(img_blurred, HTY, x_tilde, HTH, H, rho_1, rho_2, ell):
    a = rho_1
    b = rho_2
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while np.abs(c - d) > ell:
        x_c = np.real(np.fft.ifftn((HTY + c * np.fft.fftn(x_tilde)) / (HTH + c)))
        r_c = (img_blurred - np.real(np.fft.ifftn(H * np.fft.fftn(x_c))))
        m_r_c = measure(r_c)
        x_d = np.real(np.fft.ifftn((HTY + d * np.fft.fftn(x_tilde)) / (HTH + d)))
        r_d = (img_blurred - np.real(np.fft.ifftn(H * np.fft.fftn(x_d))))
        m_r_d = measure(r_d)
        if m_r_c < m_r_d:
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    rho = (b + a) / 2.0
    x = np.real(np.fft.ifftn((HTY + rho * np.fft.fftn(x_tilde)) / (HTH + rho)))
    return x, rho

def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fft2(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

