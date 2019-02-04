import time
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cpu')
np.random.seed(1001)

class Network(nn.Module):
    def __init__(self, num_maps = 3):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.pool =  nn.AvgPool2d(kernel_size = (28, 28))
        self.fc_1 = nn.Linear(num_maps, 10, bias = False)
        self.num_maps = num_maps

    def forward(self, x, labels):
        conv_1 = F.relu(self.conv1(x))
        conv_2 = F.relu(self.conv2(conv_1))
        conv_3 = F.relu(self.conv3(conv_2))
        conv_4 = F.relu(self.conv4(conv_3))        
        conv_5 = F.relu(self.conv5(conv_4))
        pool = self.pool(conv_5)
        flat = pool.view(-1, num_maps)
        linear = self.fc_1(flat)
        y = F.log_softmax(linear, dim = 1)
        recons = F.relu(torch.sum(network.fc_1.weight[labels].view(-1, self.num_maps, 1, 1) * conv_5, dim = 1))
        recons = recons / (torch.max(recons.view(-1, 28 * 28), dim = 1)[0].view(-1, 1, 1) + 0.00001)
        return y, recons, [conv_5, flat]

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:])


def trainTest(data, batch_size, network, criterion, optimizer, is_train = True):
    data_X, data_Y = data
    pred_Y = np.zeros_like(data_Y)
    running_loss = 0.0
    num_batches = int(data_X.shape[0] / batch_size)
    for batch in range(num_batches):
        # get the inputs
        start, end = batch * batch_size, (batch + 1) * batch_size
        inputs, labels = data_X[start : end].to(device), data_Y[start : end].to(device).long()
        # zero the parameter gradients
        if is_train == True:
            optimizer.zero_grad()

        probs, recons, feats = network(inputs, labels)

        pred_Y[start : end] = np.argmax(probs.cpu().detach().numpy().copy(), axis = 1)
        if is_train == True:
            loss = criterion[0](probs, labels) + 0.1 * torch.norm(network.fc_1.weight, 1) + 0.001 * criterion[1](inputs, recons)
        else:
            loss = criterion[0](probs, labels)
        if is_train == True:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / data_X.shape[0], pred_Y

def getAcc(pred, truth):
    #print(pred)
    return float(np.sum(pred == truth)) / pred.shape[0] * 100

num_maps = 3
network = Network(num_maps).to(device)
print(network)
print('Number of parameters = ', sum(p.numel() for p in network.parameters() if p.requires_grad))

criterion = [nn.NLLLoss(reduction = 'sum'), nn.MSELoss(reduction = 'sum')]
lr = 0.001
optimizer = optim.Adam(network.parameters(), lr = lr)

print('Load data')
train_X, train_Y = loadData('data/train.h5')
valid_X, valid_Y = loadData('data/valid.h5')
test_X, test_Y = loadData('data/test.h5')

epochs = 50
batch_size = 50

test_loss, _ = trainTest((test_X, test_Y), batch_size, network, criterion, optimizer, is_train = False)
print('Test loss for random basline = {}'.format(test_loss))

MODE = 'TRAIN'
finetune = False

if 'TRAIN' in MODE:
    if finetune == True:
        network.load_state_dict(torch.load('models/network_sparse_interpretable_10_'))
        print('Finetuning network')
    # Training
    loss_history = [np.inf]
    patience, impatience, limit = 5, 0, 3
    best_epoch, best_valid_loss = 0, np.inf
    for epoch in range(epochs):

        valid_loss, valid_pred = trainTest((valid_X, valid_Y), batch_size, network, criterion, optimizer, is_train = False)
        train_loss, train_pred = trainTest((train_X, train_Y), batch_size, network, criterion, optimizer, is_train = True)
        train_acc, valid_acc = getAcc(train_pred, train_Y.numpy()), getAcc(valid_pred, valid_Y.numpy())


        print('Epoch {}, Training-Loss = {}, Valid-Loss = {}'.format(epoch, train_loss, valid_loss))
        print('Epoch {}, Training-acc = {}, Valid-acc = {}'.format(epoch, train_acc, valid_acc))

        impatience += 1
        if valid_loss < min(loss_history):
            print('A better model has been obtained. Saving this model to models/network_sparse_interpretable_10')
            torch.save(network.state_dict(), 'models/network_sparse_interpretable_10')
            best_loss, best_epoch = valid_loss, epoch + 1
            impatience = 0
        loss_history.append(valid_loss)
        if impatience == patience:
            impatience = 0
            if limit != 0:
                lr /= 2
                optimizer = optim.Adam(network.parameters(), lr = lr)
                limit -= 1
                network.load_state_dict(torch.load('models/network_sparse_interpretable_10'))
            else:
                break
    print('Finished Training: best model at {} epochs with valid loss = {}'.format(best_epoch, best_loss))

if 'TEST' in MODE:
    # Testing
    network.load_state_dict(torch.load('models/network_sparse_interpretable_10', map_location = lambda storage, loc : storage))
    print('Loaded model, starting prediction')
    start = time.time()
    batch_size = 100
    test_loss, pred_Y = trainTest((test_X, test_Y), batch_size, network, criterion, optimizer, is_train = False)
    end = time.time()
    test_Y, pred_Y = np.int32(test_Y), np.int32(pred_Y)
    print('Accuracy = {}%'.format(np.sum(pred_Y == test_Y) / test_Y.shape[0] * 100))
    print('Test loss = {}'.format(test_loss))
    print('Time taken (ms) = {}'.format((end - start) / test_Y.shape[0] * 1000))
    with open('models/results_sparse_square.txt', 'w') as f:
        for index in range(test_Y.shape[0]):
            f.write('{},{}\n'.format(int(test_Y[index]), int(pred_Y[index])))
