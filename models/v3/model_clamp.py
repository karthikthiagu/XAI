import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self, num_maps):
        super(Network, self).__init__()
        self.num_maps = num_maps
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = self.num_maps, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.pool = nn.AvgPool2d((16, 16))
        self.fc_1 = nn.Linear(self.num_maps, 2, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        #print(conv_1.size())
        conv_2 = F.relu(self.conv2(conv_1))
        #print(conv_2.size())
        conv_3 = F.relu(self.conv3(conv_2))
        #print(conv_3.size())
        max_clamp = 0.75
        maps_max = conv_3.view(-1, self.num_maps, 16 * 16).max(dim = 2)[0].view(-1, self.num_maps, 1, 1) * max_clamp
        #print(maps_max.size())
        conv_3[np.where(conv_3.detach().cpu().numpy() < maps_max.detach().cpu().numpy())] = 0
        pool = self.pool(conv_3)
        #print(pool.size())
        flat = pool.view(-1, self.num_maps)
        #print(flat.size())
        y = F.log_softmax(self.fc_1(flat), dim = 1)
        return y, [conv_3, flat]

