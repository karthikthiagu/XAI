import shutil
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch

from argparser import parseArguments

from model import Network

device = torch.device('cpu')
np.random.seed(1001)

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:]).long()

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

################# Argument parsing ########################
###########################################################
args = parseArguments()
test_file = args.test_file
load_model = args.load_model
num_maps = args.num_maps
plot_folder = args.plot_folder

######## Plot folder setup #####################
if os.path.isdir(plot_folder):
    shutil.rmtree(plot_folder)
os.mkdir(plot_folder)
os.mkdir(os.path.join(plot_folder, '0'))
os.mkdir(os.path.join(plot_folder, '1'))

#################  Network definition #####################
###########################################################
network = Network(num_maps).to(device)
print(network)
print('Number of parameters = ', sum(p.numel() for p in network.parameters() if p.requires_grad))
network.load_state_dict(torch.load(load_model, map_location = lambda storage, loc : storage))
print('Loaded model')
criterion = torch.nn.NLLLoss(reduction = 'sum')

#################  Data loading ##########################
##########################################################
print('Load data')
test_X, test_Y = loadData(test_file)
print(test_X.size(), test_Y.size())
imsize = 128
feat_size = imsize // 8

################ Adversarial example #####################
##########################################################

def generate(orig_image, label, network):

    flag = 0
    for epsilon in [0.001, 0.005, 0.01, 0.05, 0.1]:
        orig_image.requires_grad = True
        output, _ = network(orig_image.view(1, 1, 128, 128))
        prob = np.exp(output[0].detach().cpu().numpy())
        o_pred = np.argmax(prob)

        loss = criterion(output, label.view(1))
        network.zero_grad()
        loss.backward()
        data_grad = orig_image.grad.data

        perturbed_image = fgsm_attack(orig_image, epsilon, data_grad)
        
        output, _ = network(perturbed_image.view(1, 1, 128, 128))
        prob = np.exp(output[0].detach().cpu().numpy())
        p_pred = np.argmax(prob)

        if o_pred == p_pred:
            continue
        else:
            flag = 1
            break
    if flag == 0:
        return (0, None, None)
    else:
        return (1, perturbed_image, epsilon)

def plot(image, network, axes, plot_index):
    prob, feats = network(image.view(1, 1, imsize, imsize))
    prob = np.exp(prob[0].detach().cpu().numpy())
    pred = np.argmax(prob)

    # Get intermediate activations and features
    feat = feats[0].view((num_maps, feat_size, feat_size)).detach().cpu().numpy()
    flat = feats[1].view((num_maps)).detach().cpu().numpy()
    W = network.state_dict()['fc_1.weight'].detach().cpu().numpy()

    # Plot maps
    for m, axis in enumerate(axes):
        mplot = np.copy(feat[m])
        mplot = mplot - np.min(mplot)
        mplot = cv2.resize(mplot / (np.max(mplot) + 0.00001) * 255.0, (imsize, imsize))
        axis.imshow(mplot, cmap = 'jet')
        axis.set_title('w = {:.2f}, a = {:.2f}'.format(W[plot_index, m], flat[m]))

    return pred


for index in range(test_X.shape[0])[:100]:
    o_image = test_X[index]
    label = test_Y[index]
    flag, p_image, epsilon = generate(test_X[index].view(1, 1, imsize, imsize), label.view(1), network)
    if flag == 0:
        continue

    plot_indices = [0, 1]
    for plot_index in plot_indices:
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols)
        fig.set_size_inches(10, 10)
        for i in range(rows):
            for j in range(cols):
                axes[i, j].axis('off')
        o_image_ = o_image.detach().cpu().numpy().reshape((imsize, imsize)) * 255.0
        p_image_ = p_image.detach().cpu().numpy().reshape((imsize, imsize)) * 255.0

        axes[0, 0].imshow(o_image_, cmap = 'gray')
        axes[0, 2].imshow(p_image_, cmap = 'gray')

        o_pred = plot(o_image, network, [axes[1, 0], axes[1, 1], axes[1, 2]], plot_index)
        p_pred = plot(p_image, network, [axes[2, 0], axes[2, 1], axes[2, 2]], plot_index)
        equation = r'$x + \epsilon sign(\nabla_{x} J(y, f(x)))$'
        plt.suptitle('FSGM attack\n{}\n\nLabel = {}, Original prediction = {}, Adversarial prediction = {}\nepsilon = {}'.format(equation, label.cpu().detach().numpy(), o_pred, p_pred, epsilon))
        plot_path = '{}/{}/{}.png'.format(plot_folder, plot_index, index)
        print(plot_path)
        plt.savefig(plot_path)
        plt.close()
