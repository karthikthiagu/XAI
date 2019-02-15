import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, num_maps):
        super(Network, self).__init__()
        self.num_maps = num_maps
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 2, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 1, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 2, padding = 1, bias = False)
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 1, padding = 1, bias = False)
        self.conv6 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 2, padding = 1, bias = False)
        self.conv7 = nn.Conv2d(in_channels = 16, out_channels = self.num_maps, kernel_size = 5, stride = 1, padding = 1, bias = False)
        self.pool = nn.AvgPool2d((16, 16))
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
        conv_5 = F.relu(self.conv5(conv_4))
        #print(conv_5.size())
        conv_6 = F.relu(self.conv6(conv_5))
        #print(conv_6.size())
        conv_7 = F.relu(self.conv7(conv_6))
        #print(conv_7.size())
        pool = self.pool(conv_7)
        #print(pool.size())
        flat = pool.view(-1, self.num_maps)
        #print(flat.size())
        y = F.log_softmax(self.fc_1(flat), dim = 1)
        return y, [conv_6, flat]

