import shutil
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
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 2, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.pool = nn.MaxPool2d((16, 16))
        self.fc_1 = nn.Linear(2, 2, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        #print(conv_1.size())
        conv_2 = F.relu(self.conv2(conv_1))
        #print(conv_2.size())
        conv_3 = F.relu(self.conv3(conv_2))
        #print(conv_3.size())
        pool = self.pool(conv_3)
        #print(pool.size())
        flat = pool.view(-1, 2)
        #print(flat.size())
        y = F.log_softmax(self.fc_1(flat), dim = 1)
        return y, [conv_3, flat]

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:])

device = torch.device('cpu')
np.random.seed(1001)

network = Network().to(device)
network.load_state_dict(torch.load('models/v4.5/classifier', map_location = lambda storage, loc : storage))
print(network.fc_1.weight)

test_X, test_Y = loadData('data/test.h5')
test_Y = np.int32(test_Y)


base = 'models/v4.5/plots'
if os.path.isdir(base):
    shutil.rmtree(base)
os.mkdir(base)
os.mkdir('models/v4.5/plots/0')
os.mkdir('models/v4.5/plots/1')

imsize = 128
feat_size = imsize // 8
font = cv2.FONT_HERSHEY_SIMPLEX

for index in range(test_X.shape[0]):
    label = test_Y[index]
    # Get predictions
    prob, feats = network(test_X[index].view(1, 2, imsize, imsize))
    prob = np.exp(prob[0].detach().numpy())
    pred = np.argmax(prob)
    #print('prob, pred : ', prob, pred)

    # Get intermediate activations and features
    #print(feats[0].size(), feats[1].size())
    feat = feats[0].view((2, feat_size, feat_size)).detach().numpy()
    flat = feats[1].view((2)).detach().numpy()
    W = network.state_dict()['fc_1.weight'].detach().numpy()
    #print('W', W.shape)


    plot_indices = [0, 1]
    for plot_index in plot_indices:
        # Get CAM
        cam = np.zeros((feat_size, feat_size))
        for i in range(feat.shape[0]):
            cam += W[plot_index, i] * feat[i]
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 0.00001) * 255.0
        cam = cv2.resize(cam, (imsize, imsize))

        # Plot maps
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols)
        for i in range(rows):
            for j in range(cols):
                axes[i, j].axis('off')
        fig.set_size_inches(10, 10)
        left_right = test_X[index].numpy().reshape((2, imsize, imsize)) * 255.0
        text = 'Label = {}, Prediction = {}\n Prob for class {} = {:.4f}\n Map for class {}'.format(label, pred, plot_index, prob[plot_index], plot_index)
        plt.suptitle(text)
        axes[0, 0].imshow(left_right[0], cmap = 'gray')
        axes[0, 1].imshow(left_right[1], cmap = 'gray')
        axes[0, 2].imshow(cam, cmap = 'jet')

        for m in range(2):
            mplot = np.copy(feat[m])
            mplot = mplot - np.min(mplot)
            mplot = mplot / (np.max(mplot) + 0.00001) * 255.0
            rp, cp = 1, m % 2
            axes[rp, cp].imshow(mplot, cmap = 'jet')
            axes[rp, cp].set_title('w = {:.2f}, a = {:.2f}'.format(W[plot_index, m], flat[m]))
        plot_path = os.path.join('models/v4.5/plots/{}/{}.png'.format(plot_index, index))
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()
