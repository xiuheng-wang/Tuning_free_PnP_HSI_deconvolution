import torch.utils.data as data
import torch
import glob
import numpy as np
import torchvision.transforms as transforms
import tifffile as tiff
import time

def get_noised(img, sigma_min, sigma_max): 
    dim = np.shape(img)
    sigma = np.array(np.random.uniform(low=sigma_min, high=sigma_max))
    noise = sigma * np.random.randn(dim[0], dim[1], dim[2]) 
    img_noised = img + noise * 255
    img_noised = np.clip(img_noised, 0, 255)
    return img_noised.astype(np.uint8), sigma.astype(np.float32)

def flip(img, random_flip):
    # input:3D array to flip
    # output:3D array flipped
    channel = np.shape(img)[0]
    # Random horizontal flipping
    if random_flip[0] > 0.5:
        for i in range(channel):
            img[i, :, :] = np.fliplr(img[i, :, :])
    # Random vertical flipping
    if random_flip[1] > 0.5:
        for i in range(channel):
            img[i, :, :] = np.flipud(img[i, :, :])
    # Rotate img by 90 degrees
    if random_flip[2] > 0.5:
        for i in range(channel):
            img[i, :, :] = np.rot90(img[i, :, :])
    return img

class Dataset_cave_train(data.Dataset):
	def __init__(self, file_train_path, sigma_min, sigma_max):
		self.tif_list = glob.glob(file_train_path+'/*.tif')
		self.transforms = transforms.ToTensor()
		self.sigma_min = sigma_min
		self.sigma_max = sigma_max

	def __getitem__(self, index):
		index = index % len(self.tif_list)
		img = tiff.imread(self.tif_list[index]).astype(np.uint8) # [0, 255]
		patch_size = 64 
		
		np.random.seed(int(time.time()*1000)%1000000)
		topleft = np.random.random_integers(0, 512-patch_size, 2)
		img = img[:, topleft[0]:topleft[0]+patch_size, topleft[1]:topleft[1]+patch_size]

		random_flip = np.random.random(3)
		img = flip(img, random_flip)

		img_noised, sigma = get_noised(img, self.sigma_min, self.sigma_max)

		data_hsi = self.transforms(img_noised.transpose([1, 2, 0]))
		label = self.transforms(img.transpose([1, 2, 0]))

		sigma = torch.from_numpy(sigma)

		return data_hsi.unsqueeze(0), label.unsqueeze(0), sigma

	def __len__(self):
		return 128 * len(self.tif_list) # num of patchs is 50 in one raw train image

class Dataset_cave_val(data.Dataset):
	def __init__(self, file_val_path, sigma_min, sigma_max):
		self.tif_list = glob.glob(file_val_path+'/*.tif')
		self.transforms = transforms.ToTensor()
		self.sigma_min = sigma_min
		self.sigma_max = sigma_max

	def __getitem__(self, index):
		index = index % len(self.tif_list)
		img = tiff.imread(self.tif_list[index]).astype(np.uint8) # [0, 255]
		patch_size = 64
		
		np.random.seed(int(time.time()*1000)%1000000)
		topleft = np.random.random_integers(0, 512-patch_size, 2)
		img = img[:, topleft[0]:topleft[0]+patch_size, topleft[1]:topleft[1]+patch_size]

		random_flip = np.random.random(3)
		img = flip(img, random_flip)

		img_noised, sigma = get_noised(img, self.sigma_min, self.sigma_max)

		data_hsi = self.transforms(img_noised.transpose([1, 2, 0]))
		label = self.transforms(img.transpose([1, 2, 0]))

		sigma = torch.from_numpy(sigma)

		return data_hsi.unsqueeze(0), label.unsqueeze(0), sigma

	def __len__(self):
		return 16 * len(self.tif_list)

class Dataset_cave_test(data.Dataset):
	def __init__(self, file_test_path, sigma):
		self.tif_list = glob.glob(file_test_path+'/*.tif')
		self.transforms = transforms.ToTensor()
		self.sigma = sigma

	def __getitem__(self, index):
		index = index % len(self.tif_list)
		img = tiff.imread(self.tif_list[index]).astype(np.uint8) # [0, 255]

		img_noised, sigma = get_noised(img, self.sigma, self.sigma)

		data_hsi = self.transforms(img_noised.transpose([1, 2, 0]))
		label = self.transforms(img.transpose([1, 2, 0]))

		sigma = torch.from_numpy(sigma)

		return data_hsi.unsqueeze(0), label.unsqueeze(0), sigma

	def __len__(self):
		return 1 * len(self.tif_list)
