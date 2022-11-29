# coding: utf-8
# Script for training B3DDN
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
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse

import  os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import time

from dataset import Dataset_cave_train, Dataset_cave_val
from DnCNN import Net_0
from utils import AverageMeter,initialize_logger,save_checkpoint,record_loss
from loss import Image_Loss


# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='CAVE database superresolution:')
# model storage path
parser.add_argument('--sigma_min', type=float, default=0.2 / 255.0, help='Set minimum standard deviation of Gaussian noise ')
parser.add_argument('--sigma_max', type=float, default=10 / 255.0, help='Set maximum standard deviation of Gaussian noise ')
# model configuration
parser.add_argument('--num_blocks', type=int, default=8, help="Set numbers of 3D residual blocks in the noise estimation network")
parser.add_argument('--num_kernels', type=int, default=32, help="Set numbers of 3D kernels in the noise estimation network")
# model storage path
parser.add_argument('--model_path', type=str, default='./models/', help='Set model storage path')

args = parser.parse_args()

def main():

    cudnn.benchmark = True 
    train_data = Dataset_cave_train('./data/train', args.sigma_min, args.sigma_max)
    print('number of train data: ', len(train_data))
    val_data = Dataset_cave_val('./data/train', args.sigma_min, args.sigma_max)
    print('number of validate data: ', len(val_data))

    # Model               
    model = Net_0(num_blocks=args.num_blocks, num_kernels=args.num_kernels)

    # multi-GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)

    model=model.to(device=device, dtype=torch.float) # float32
    model.apply(weights_init_kaiming)

    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 501
    init_lr = 0.0002
    iteration = 0

    criterion = Image_Loss() 
    optimizer=torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    model_path = args.model_path 
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss.csv'), 'w+')
    
    log_dir = os.path.join(model_path,'train.log')
    logger = initialize_logger(log_dir)

    resume_file = ''   
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch+1, end_epoch):

        train_data = Dataset_cave_train('./data/train', args.sigma_min, args.sigma_max)
        train_data_loader = DataLoader(dataset=train_data, 
                                   num_workers=8,  
                                   batch_size=128,
                                   shuffle=True,
                                   pin_memory=True,
                                   )
        
        val_data = Dataset_cave_val('./data/train', args.sigma_min, args.sigma_max)
        val_data_loader = DataLoader(dataset=val_data,
                            num_workers=8, 
                            batch_size=128,
                            shuffle=False,
                            pin_memory=True)

        start_time = time.time()
        train_loss, iteration = train(train_data_loader, model, criterion, optimizer, iteration, device)
        val_loss = validate(val_data_loader, model, criterion, device)

        # Save model
        if epoch % (end_epoch-1) ==0:
            save_checkpoint(model_path, epoch, iteration, model, optimizer)
        
        # # Update learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        # scheduler.step(val_loss)

        # print loss 
        end_time = time.time()
        epoch_time = end_time - start_time
        print ("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, val_loss))
        # save loss
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, val_loss)   # 调用record_方法：将epoch等6个指标写到csv文件中  
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, val_loss))


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

# Training 
def train(train_data_loader, model, criterion, optimizer, iteration, device):
    losses = AverageMeter()
    for i, (data_hsi, label, label_sigma) in enumerate(train_data_loader):

        data_hsi = data_hsi.to(device=device) 
        label = label.to(device=device)
        label_sigma = label_sigma.to(device=device)
        
        iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(data_hsi)
        # print(sigma.size())
        # print(label_sigma.size())
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        #  record loss
        losses.update(loss.item()) 
            
    return losses.avg, iteration 

# Validate
def validate(val_data_loader, model, criterion, device):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (data_hsi, label, label_sigma) in enumerate(val_data_loader):

        data_hsi = data_hsi.to(device=device) 
        label = label.to(device=device)
        label_sigma = label_sigma.to(device=device)
        
        output = model(data_hsi)
        loss = criterion(output, label)

        #  record loss
        losses.update(loss.item())

    return losses.avg


if __name__ == '__main__':
    main()
