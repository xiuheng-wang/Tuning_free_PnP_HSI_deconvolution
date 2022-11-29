import torch
from torch import nn

class Net_0(nn.Module):
    def __init__(self, num_blocks, num_kernels):
        super(Net_0, self).__init__()
        self.input = nn.Sequential(
            nn.Conv3d(1, num_kernels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True))
        self.hidden_layers = nn.Sequential(*[nn.Sequential(P3D_block(num_kernels)) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv3d(num_kernels, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)))

    def forward(self, input_hsi):

        out1 = self.input(input_hsi)
        out2 = self.hidden_layers(out1)
        out3 = self.output(out2)

        return torch.add(input_hsi, out3)

class P3D_block(nn.Module):
    def __init__(self, num_kernels):
        super(P3D_block, self).__init__()
        self.conv = nn.Conv3d(num_kernels, num_kernels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(num_kernels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
