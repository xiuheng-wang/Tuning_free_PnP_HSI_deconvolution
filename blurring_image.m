% Script for getting blurred images
%
% Reference: 
% Tuning-free Plug-and-Play Hyperspectral Image Deconvolution with Deep Priors
% Xiuheng Wang, Jie Chen, C¨¦dric Richard, David Brie
%
% 2021/03
% Implemented by
% Xiuheng Wang
% xiuheng.wang@oca.eu

clear;clc;
close all;

folderTest   = 'data/test/';
folderKernel  = 'data/kernels/';

img_blurred = zeros(31, 512, 512);
for i = 1:6
    folderResult = fullfile(strcat( 'data/blurred/blurred_', num2str(i), '/'));
    if ~exist(folderResult,'file')
        mkdir(folderResult);
    end  
    if i == 3
        sigma = 0.03;
    else
        sigma = 0.01;
    end
    kernel = load(fullfile(strcat( folderKernel, 'kernel_', int2str(i), '.mat' )));
    kernel = kernel.kernel;
    
    for j = 1:12
        load(fullfile(strcat( folderTest, int2str(j-1), '.mat' )));
        img_blurred = imfilter(im2double(permute(img, [2, 3, 1])), kernel, 'circular', 'conv');
        img_blurred = permute(img_blurred, [3, 1, 2]) + sigma * randn(size(img));
        save(strcat(folderResult, int2str(j-1), '.mat'), 'img_blurred');
    end
end
