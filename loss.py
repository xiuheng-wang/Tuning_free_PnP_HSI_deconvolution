import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Image_Loss(nn.Module):
    def __init__(self):
        super(Image_Loss, self).__init__()
        self.image_loss = nn.L1Loss()

    def forward(self, output, label):
        image_loss = self.image_loss(output, label)
        return image_loss