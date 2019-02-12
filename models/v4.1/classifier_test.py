import time
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparser import parseArguments

device = torch.device('cuda')
np.random.seed(1001)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 9, stride = 2, padding = 4, bias = False)
        self.pool = nn.MaxPool2d((16, 16))
        self.fc_1 = nn.Linear(8, 2, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        #print(conv_1.size())
        conv_2 = F.relu(self.conv2(conv_1))
        #print(conv_2.size())
        conv_3 = F.relu(self.conv3(conv_2))
        #print(conv_3.size())
        pool = self.pool(conv_3)
        #print(pool.size())
        flat = pool.view(-1, 8)
        #print(flat.size())
        y = F.log_softmax(self.fc_1(flat), dim = 1)
        #print(self.fc_1.weight.requires_grad)
        #print(self.fc_1.weight)
        #print(y.size())
        return y, flat

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:]).long()

def trainTest(data, batch_size, network, criterion):
    data_X, data_Y = data
    pred_Y = np.zeros_like(data_Y)
    running_loss = 0.0
    num_batches = int(data_X.shape[0] / batch_size)
    for batch in range(num_batches):
        # get the inputs
        start, end = batch * batch_size, (batch + 1) * batch_size
        inputs, labels = data_X[start : end].to(device), data_Y[start : end].to(device)

        outputs, feats = network(inputs)

        pred_Y[start : end] = np.argmax(outputs.cpu().detach().numpy().copy(), axis = 1)

        loss = criterion(outputs, labels) #+ 0.01 * torch.norm(feats, 1)

        running_loss += loss.item()

    return running_loss / data_X.shape[0], pred_Y

def getAcc(pred, truth):
    return np.float32(np.sum(np.int32(pred) == np.int32(truth))) / pred.shape[0] * 100


################# Argument parsing ########################
###########################################################
args = parseArguments()
batch_size = args.batch_size
test_file = args.test_file
load_model = args.load_model
results_file = args.save_results

#################  Network definition #####################
###########################################################
network = Network().to(device)
print(network)
print('Number of parameters = ', sum(p.numel() for p in network.parameters() if p.requires_grad))
criterion = nn.NLLLoss(reduction = 'sum')


#################  Data loading ##########################
##########################################################
print('Load data')
test_X, test_Y = loadData(test_file)
print(test_X.size(), test_Y.size())

################# Testing for random baseline ##############
############################################################
test_loss, _ = trainTest((test_X, test_Y), batch_size, network, criterion)
print('Test loss for random basline = {}'.format(test_loss))

################# Training #################################
############################################################
network.load_state_dict(torch.load(load_model, map_location = lambda storage, loc : storage))
print('Loaded model, starting prediction')
start = time.time()
test_loss, pred_Y = trainTest((test_X, test_Y), batch_size, network, criterion)
end = time.time()
test_Y, pred_Y = np.int32(test_Y), np.int32(pred_Y)
print('Accuracy = {}%'.format(np.sum(pred_Y == test_Y) / test_Y.shape[0] * 100))
print('Test loss = {}'.format(test_loss))
print('Time taken (ms) = {}'.format((end - start) / test_Y.shape[0] * 1000))
with open(results_file, 'w') as f:
    for index in range(test_Y.shape[0]):
        f.write('{},{}\n'.format(int(test_Y[index]), int(pred_Y[index])))
