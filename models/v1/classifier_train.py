import time
import h5py
import numpy as np

import torch

from argparser import parseArguments

from model import Network

device = torch.device('cuda')
np.random.seed(113)

def loadData(path):
    data = h5py.File(path, 'r')
    return torch.from_numpy(data['X'][:]), torch.from_numpy(data['Y'][:]).long()

def trainTest(data, batch_size, network, criterion, optimizer, is_train = True):
    data_X, data_Y = data
    pred_Y = np.zeros_like(data_Y)
    running_loss = 0.0
    num_batches = int(data_X.shape[0] / batch_size)
    for batch in range(num_batches):
        # get the inputs
        start, end = batch * batch_size, (batch + 1) * batch_size
        inputs, labels = data_X[start : end].to(device), data_Y[start : end].to(device)
        # zero the parameter gradients
        if is_train == True:
            optimizer.zero_grad()

        outputs, feats = network(inputs)

        pred_Y[start : end] = np.argmax(outputs.cpu().detach().numpy().copy(), axis = 1)

        loss = criterion(outputs, labels)
        if is_train == True:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / data_X.shape[0], pred_Y

def getAcc(pred, truth):
    return np.float32(np.sum(np.int32(pred) == np.int32(truth))) / pred.shape[0] * 100


################# Argument parsing ########################
###########################################################
args = parseArguments()
epochs, batch_size, lr = args.epochs, args.batch_size, args.lr
train_file, valid_file = args.train_file, args.valid_file
num_maps = args.num_maps
finetune = args.finetune
patience, limit = args.patience, args.limit
load_model, save_model = args.load_model, args.save_model

#################  Network definition #####################
###########################################################
network = Network(num_maps).to(device)
print(network)
print('Number of parameters = ', sum(p.numel() for p in network.parameters() if p.requires_grad))
criterion = torch.nn.NLLLoss(reduction = 'sum')
optimizer = torch.optim.Adam(network.parameters(), lr = lr)

#################  Data loading ##########################
##########################################################
print('Load data')
train_X, train_Y = loadData(train_file)
valid_X, valid_Y = loadData(valid_file)
print(train_X.size(), train_Y.size())

################# Testing for random baseline ##############
############################################################
test_loss, _ = trainTest((valid_X, valid_Y), batch_size, network, criterion, optimizer, is_train = False)
print('Test loss for random basline = {}'.format(test_loss))

################# Checking finetune conditions #############
############################################################
finetune = False if args.finetune == 'False' else True
if finetune == True:
    network.load_state_dict(torch.load(load_model))
    print('Finetuning network')

################# Training #################################
############################################################
loss_history = [np.inf]
impatience= 0
best_epoch, best_valid_loss = 0, np.inf
for epoch in range(epochs):
    valid_loss, valid_pred = trainTest((valid_X, valid_Y), batch_size, network, criterion, optimizer, is_train = False)
    train_loss, train_pred = trainTest((train_X, train_Y), batch_size, network, criterion, optimizer, is_train = True)
    train_acc, valid_acc = getAcc(train_pred, train_Y.numpy()), getAcc(valid_pred, valid_Y.numpy())

    print('Epoch {}, Training-Loss = {:.4f}, Valid-Loss = {:.4f}'.format(epoch, train_loss, valid_loss))
    print('Epoch {}, Training-acc = {:.4f}, Valid-acc = {:.4f}'.format(epoch, train_acc, valid_acc))

    impatience += 1
    if valid_loss < min(loss_history):
        print('A better model has been obtained. Saving this model to {}'.format(save_model))
        torch.save(network.state_dict(), save_model)
        best_loss, best_epoch = valid_loss, epoch + 1
        impatience = 0
    loss_history.append(valid_loss)
    if impatience == patience:
        impatience = 0
        if limit != 0:
            lr /= 2
            optimizer = torch.optim.Adam(network.parameters(), lr = lr)
            limit -= 1
            network.load_state_dict(torch.load(save_model))
        else:
            break
print('Finished Training: best model at {} epochs with valid loss = {}'.format(best_epoch, best_loss))

