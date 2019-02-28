import time
import h5py
import numpy as np

import torch

from argparser import parseArguments

from auto import Network

device = torch.device('cuda')
np.random.seed(1001)

def loadData(path):
    data = h5py.File(path, 'r')
    imsize = 128
    return torch.from_numpy(data['X'][:]).view(-1, 1, imsize, imsize), torch.from_numpy(data['Y'][:]).long()

def test(data, batch_size, network, num_maps):
    data_X, data_Y = data
    recon = np.zeros_like(data_X)
    running_loss = 0.0
    num_batches = int(data_X.shape[0] / batch_size)
    embed = np.zeros((data_X.shape[0], num_maps, 16, 16))
    for batch in range(num_batches):
        # get the inputs
        start, end = batch * batch_size, (batch + 1) * batch_size
        inputs =  data_X[start : end].to(device)

        outputs, feats = network(inputs)

        embed[start : end] = feats.cpu().detach().numpy().copy()
        recon[start : end] = outputs.cpu().detach().numpy().copy()

    return recon, embed, data_Y.numpy()

################# Argument parsing ########################
###########################################################
args = parseArguments()
batch_size = args.batch_size
test_file = args.test_file
load_model = args.load_model
num_maps = args.num_maps
results_file = args.save_results

#################  Network definition #####################
###########################################################
network = Network(num_maps).to(device)
print(network)
print('Number of parameters = ', sum(p.numel() for p in network.parameters() if p.requires_grad))

#################  Data loading ##########################
##########################################################
print('Load data')
test_X, test_Y = loadData(test_file)
print(test_X.size(), test_Y.size())

################# Training #################################
############################################################
network.load_state_dict(torch.load(load_model, map_location = lambda storage, loc : storage))
print('Loaded model, starting prediction')
recon_X, embed_X, data_Y = test((test_X, test_Y), batch_size, network, num_maps)
print(embed_X.shape, data_Y.shape)
with h5py.File(results_file, 'w') as dataset:
    dataset.create_dataset(data = data_Y, name = 'Y', shape = data_Y.shape, dtype = data_Y.dtype)
    dataset.create_dataset(data = embed_X, name = 'X', shape = embed_X.shape, dtype = embed_X.dtype)
    dataset.create_dataset(data = recon_X, name = 'rX', shape = recon_X.shape, dtype = recon_X.dtype)
