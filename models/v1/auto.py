import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self, num_maps):
        super(Network, self).__init__()
        self.num_maps = num_maps
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = self.num_maps, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.tconv3 = nn.ConvTranspose2d(in_channels = self.num_maps, out_channels = 8, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        self.tconv2 = nn.ConvTranspose2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False)
        self.tconv1 = nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        #print(conv_1.size())
        conv_2 = F.relu(self.conv2(conv_1))
        #print(conv_2.size())
        conv_3 = F.relu(self.conv3(conv_2))
        #print(conv_3.size())
        tconv_3 = F.relu(self.tconv3(conv_3))
        #print(tconv_3.size())
        tconv_2 = F.relu(self.tconv2(tconv_3))
        #print(tconv_2.size())
        x = F.relu(self.tconv1(tconv_2))
        #print(x.size())
        return x, conv_3

if __name__ == '__main__':
    input_arr = torch.tensor(np.zeros((10, 1, 128, 128), dtype = np.float32))
    net = Network(1)
    net(input_arr)

