import shutil
import os
import time
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import torch

from argparser import parseArguments

from model import Network

device = torch.device('cuda')
np.random.seed(1001)

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]).to(device), torch.from_numpy(data['Y'][:]).to(device)

############ Parse arguments ############
args = parseArguments()
load_model = args.load_model
test_file = args.test_file
plot_folder = args.plot_folder
num_maps = args.num_maps

########## Network definition ##########
network = Network(num_maps).to(device)
network.load_state_dict(torch.load(load_model, map_location = lambda storage, loc : storage))

######### Load data ###############
test_X, test_Y = loadData(test_file)
print(test_X.shape, test_Y.shape)

######## Plot folder setup
if os.path.isdir(plot_folder):
    shutil.rmtree(plot_folder)
os.mkdir(plot_folder)
os.mkdir(os.path.join(plot_folder, '0'))
os.mkdir(os.path.join(plot_folder, '1'))

####### Predict and plot ###########
imsize = 128
feat_size = imsize // 8
font = cv2.FONT_HERSHEY_SIMPLEX
for index in range(test_X.shape[0])[ : 100]:
    label = test_Y[index]
    # Get predictions
    prob, feats = network(test_X[index].view(1, 1, imsize, imsize))
    prob = np.exp(prob[0].detach().cpu().numpy())
    pred = np.argmax(prob)

    # Get intermediate activations and features
    print(feats[0].size())
    feat = feats[0].view((num_maps, feat_size, feat_size)).detach().cpu().numpy()
    flat = feats[1].view((num_maps)).detach().cpu().numpy()
    W = network.state_dict()['fc_1.weight'].detach().cpu().numpy()

    plot_indices = [0, 1]
    for plot_index in plot_indices:
        # Get CAM
        '''
        cam = np.zeros((feat_size, feat_size))
        for i in range(feat.shape[0]):
            cam += W[plot_index, i] * feat[i]
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 0.00001) * 255.0
        cam = cv2.resize(cam, (imsize, imsize))
        '''

        # Plot maps
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols)
        for i in range(rows):
            for j in range(cols):
                axes[i, j].axis('off')
        fig.set_size_inches(10, 10)
        image = test_X[index].cpu().numpy().reshape((imsize, imsize)) * 255.0
        text = 'Label = {}, Prediction = {}\n Prob for class {} = {:.4f}\n Map for class {}'.format(label, pred, plot_index, prob[plot_index], plot_index)
        plt.suptitle(text)
        axes[0, 0].imshow(image, cmap = 'gray')
        #axes[0, 1].imshow(cam, cmap = 'jet')

        for m in range(num_maps):
            mplot = np.copy(feat[m])
            mplot = mplot - np.min(mplot)
            mplot = cv2.resize(mplot / (np.max(mplot) + 0.00001) * 255.0, (imsize, imsize))
            rp, cp = m // 4 + 1, m % 4
            axes[rp, cp].imshow(mplot, cmap = 'jet')
            axes[rp, cp].set_title('w = {:.2f}, a = {:.2f}'.format(W[plot_index, m], flat[m]))

        plot_path = '{}/{}/{}.png'.format(plot_folder, plot_index, index)
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()
