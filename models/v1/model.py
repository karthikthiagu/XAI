import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self, num_maps):
        super(Network, self).__init__()
        self.num_maps = num_maps
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 9, stride = 1, padding = 4, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = self.num_maps, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.pool = nn.MaxPool2d((16, 16))
        self.fc_1 = nn.Linear(self.num_maps, 2, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        #print(conv_1.size())
        conv_2 = F.relu(self.conv2(conv_1))
        #print(conv_2.size())
        conv_3 = F.relu(self.conv3(conv_2))
        #print(conv_3.size())
        conv_4 = F.relu(self.conv4(conv_3))
        #print(conv_4.size())
        pool = self.pool(conv_4)
        #print(pool.size())
        flat = pool.view(-1, self.num_maps)
        #print(flat.size())
        y = F.log_softmax(self.fc_1(flat), dim = 1)
        return y, [conv_4, flat]

if __name__ == '__main__':
    input_arr = torch.tensor(np.zeros((10, 1, 128, 128), dtype = np.float32))
    net = Network(16)
    net(input_arr)

