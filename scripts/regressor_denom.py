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
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.pool =  nn.MaxPool2d(kernel_size = (8, 8))
        self.fc_1 = nn.Linear(16, 1, bias = False)

    def forward(self, x):
        conv_1 = F.relu(self.conv1(x))
        conv_2 = F.relu(self.conv2(conv_1))
        conv_3 = F.relu(self.conv3(conv_2))
        conv_4 = F.relu(self.conv4(conv_3))
        conv_5 = F.relu(self.conv5(conv_4))
        pool = self.pool(conv_5)
        flat = pool.view(-1, 16)
        y = self.fc_1(flat)
        return y

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
        inputs, labels = data_X[start : end].to(device), data_Y[start : end].to(device)
        #print(labels)
        # zero the parameter gradients
        if is_train == True:
            optimizer.zero_grad()

        outputs = network(inputs)
        #print(outputs.size())
        pred_Y[start : end] = outputs.squeeze().cpu().detach().numpy().copy()
        loss = criterion(outputs, labels)
        if is_train == True:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / data_X.shape[0], pred_Y

def getAcc(pred, truth):
    #print(pred)
    return np.float32(np.sum(np.int32(pred) == np.int32(truth))) / pred.shape[0] * 100

network = Network().to(device)
print(network)
print('Number of parameters = ', sum(p.numel() for p in network.parameters() if p.requires_grad))

criterion = nn.MSELoss(reduction = 'sum')
lr = 0.001
optimizer = optim.Adam(network.parameters(), lr = lr)

print('Load data')
train_X, train_Y = loadData('data/denom/train.h5')
valid_X, valid_Y = loadData('data/denom/valid.h5')
print(train_X.size(), train_Y.size())

epochs = 100
batch_size = 20

test_loss, _ = trainTest((valid_X, valid_Y), batch_size, network, criterion, optimizer, is_train = False)
print('Test loss for random basline = {}'.format(test_loss))

MODE = 'TRAIN'
finetune = False

if 'TRAIN' in MODE:
    if finetune == True:
        network.load_state_dict(torch.load('models/denom/regressor'))
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
            print('A better model has been obtained. Saving this model to models/denom/regressor')
            torch.save(network.state_dict(), 'models/denom/regressor')
            best_loss, best_epoch = valid_loss, epoch + 1
            impatience = 0
        loss_history.append(valid_loss)
        if impatience == patience:
            impatience = 0
            if limit != 0:
                lr /= 2
                optimizer = optim.Adam(network.parameters(), lr = lr)
                limit -= 1
                network.load_state_dict(torch.load('models/denom/regressor'))
            else:
                break
    print('Finished Training: best model at {} epochs with valid loss = {}'.format(best_epoch, best_loss))

if 'TEST' in MODE:
    #test_X, test_Y = valid_X, valid_Y
    test_X, test_Y = train_X, train_Y
    # Testing
    network.load_state_dict(torch.load('models/denom/regressor', map_location = lambda storage, loc : storage))
    print('Loaded model, starting prediction')
    start = time.time()
    batch_size = 10
    test_loss, pred_Y = trainTest((test_X, test_Y), batch_size, network, criterion, optimizer, is_train = False)
    end = time.time()
    test_Y, pred_Y = np.int32(test_Y), np.int32(pred_Y)
    print('Accuracy = {}%'.format(np.sum(pred_Y == test_Y) / test_Y.shape[0] * 100))
    print('Test loss = {}'.format(test_loss))
    print('Time taken (ms) = {}'.format((end - start) / test_Y.shape[0] * 1000))
    with open('models/denom/results_regressor.txt', 'w') as f:
        for index in range(test_Y.shape[0]):
            f.write('{},{}\n'.format(int(test_Y[index]), int(pred_Y[index])))
