import time
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:])

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias  = False)
        self.pool =  nn.AvgPool2d(kernel_size = (7, 7))
        self.fc_1 = nn.Linear(32, 10, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        #print(x.size())
        conv_2 = F.relu(self.conv2(conv_1))
        #print(x.size())
        conv_3 = F.relu(self.conv3(conv_2))
        #print(x.size())        
        pool = self.pool(conv_3)
        #print(x.size())
        flat = pool.view(-1, 32)
        #print(x.size())
        linear = self.fc_1(flat)
        #print(x.size())
        y = F.log_softmax(linear, dim = 1)
        #print(x.size())
        return y, [conv_3, flat]


device = torch.device('cpu')
np.random.seed(1001)

mean, sigma = 0.1307, 0.3081

network = Network().to(device)
network.load_state_dict(torch.load('models/network', map_location = lambda storage, loc : storage))

test_X, test_Y = loadData('data/test.h5')
test_Y = np.int32(test_Y)

index = 0
prob_y, feats = network(test_X[index].view(1, 1, 28, 28))
pred_y = np.argmax(prob_y[0].detach().numpy())

print('Ground truth : ', test_Y[index], 'Prediction : ', pred_y)
feat = feats[0].view((32, 7, 7)).detach().numpy()
W = network.state_dict()['fc_1.weight'].detach().numpy()
print('Features : {}, Weights : {}'.format(feat.shape, W.shape))

cam = np.zeros((7, 7))
for i in range(feat.shape[0]):
    print(W[pred_y, i], feats[1][0][i].detach().numpy())
    cam += W[pred_y, i] * feat[i]

cam_res = cv2.resize(cam, (28, 28))
cam_res = cam_res - np.min(cam_res)
cam_res = cam_res / np.max(cam_res) * 255.0
plt.subplot(121)
plt.imshow(test_X[index].numpy().reshape((28, 28)) * sigma + mean, cmap = 'gray')
plt.subplot(122)
plt.imshow(cam_res, cmap = 'gray')
#sns.heatmap(cam_res)
plt.show()
