import os
import time
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, num_maps):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = num_maps, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.pool =  nn.AvgPool2d(kernel_size = (28, 28))
        self.fc_1 = nn.Linear(num_maps, 20, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        conv_2 = F.relu(self.conv2(conv_1))
        conv_3 = F.relu(self.conv3(conv_2))
        conv_4 = F.relu(self.conv4(conv_3))        
        conv_5 = F.relu(self.conv5(conv_4))        
        pool = self.pool(conv_5)
        num_maps = pool.size()[1]
        flat = pool.view(-1, num_maps)
        linear = self.fc_1(flat)
        y = F.log_softmax(linear, dim = 1)
        return y, [conv_5, flat]

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:])

device = torch.device('cpu')
np.random.seed(1001)

mean, sigma = 0.1307, 0.3081

num_maps = 10
network = Network(num_maps).to(device)
network.load_state_dict(torch.load('models/network_sparse_square_{}'.format(num_maps), map_location = lambda storage, loc : storage))

test_X, test_Y = loadData('data/test.h5')
test_Y = np.int32(test_Y)


base = 'models/{}'.format(num_maps)
if not os.path.isdir(base):
    os.mkdir(base)
for l in range(10):
    label_dir = os.path.join(base, '{}'.format(l))
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)
        os.mkdir(os.path.join(label_dir, 'T'))
        os.mkdir(os.path.join(label_dir, 'F'))

for index in range(test_X.shape[0])[:300]:
    label = test_Y[index]
    # Get predictions
    prob, feats = network(test_X[index].view(1, 1, 28, 28))
    prob = np.exp(prob[0].detach().numpy())
    pred = np.argmax(prob)

    # Get intermediate activations and features
    feat = feats[0].view((num_maps, 28, 28)).detach().numpy()
    flat = feats[1].view((num_maps)).detach().numpy()
    W = network.state_dict()['fc_1.weight'].detach().numpy()

    # Get CAM
    cam = np.zeros((28, 28))
    for i in range(feat.shape[0]):
        cam += W[pred, i] * feat[i]
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) * 255.0

    # Plot maps
    plot_count = num_maps + 2
    rows, cols = plot_count // 2 + 1 if plot_count % 2!= 0 else plot_count // 2, 2
    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(10, 10)
    axes[0, 0].imshow(test_X[index].numpy().reshape((28, 28)) * sigma + mean, cmap = 'gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Label = {}, Prediction = {}'.format(label, pred))
    axes[0, 1].imshow(cam, cmap = 'jet')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Prob = {:.2f}'.format(prob[pred]))
    for m in range(num_maps):
        mplot = np.copy(feat[m])
        mplot = mplot - np.min(mplot)
        mplot = mplot / (np.max(mplot) + 0.00001) * 255.0
        rp, cp = int(m / 2) + 1, m % 2
        axes[rp, cp].imshow(mplot, cmap = 'jet')
        axes[rp, cp].axis('off')
        axes[rp, cp].set_title('w = {:.2f}, a = {:.2f}'.format(W[pred, m], flat[m]))
    if num_maps % 2 != 0:
        fig.delaxes(axes[rows - 1, cols - 1])
    if label == pred:
        plot_path = os.path.join('models/{}/{}/T/{}.png'.format(num_maps, label, index))
    else:
        plot_path = os.path.join('models/{}/{}/F/{}.png'.format(num_maps, label, index))
    print(plot_path)
    plt.savefig(plot_path)
    plt.close()
    #plt.show()
