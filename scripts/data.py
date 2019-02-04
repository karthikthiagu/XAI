import struct
import numpy as np
import h5py

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

train_val_X, train_val_Y = read_idx('data/raw/train-images-idx3-ubyte'), read_idx('data/raw/train-labels-idx1-ubyte')
test_X, test_Y = read_idx('data/raw/t10k-images-idx3-ubyte'), read_idx('data/raw/t10k-labels-idx1-ubyte')
indices = np.arange(train_val_Y.shape[0])

np.random.seed(1729)
np.random.shuffle(indices)

train_X, train_Y = train_val_X[indices[ : 50000]], train_val_Y[indices[ : 50000]]
valid_X, valid_Y = train_val_X[indices[-10000 : ]], train_val_Y[indices[-10000 : ]]
#mean, sigma = 0.1307, 0.3081
mean, sigma = 0.0, 255.0
train_X, train_Y = ((np.float32(train_X) - mean) / sigma).reshape(-1, 1, 28, 28), np.float32(train_Y)
valid_X, valid_Y = ((np.float32(valid_X) - mean) / sigma).reshape(-1, 1, 28, 28), np.float32(valid_Y)
test_X, test_Y = ((np.float32(test_X) - mean) / sigma).reshape(-1, 1, 28, 28), np.float32(test_Y)

print('Train images : ', train_X.shape, train_Y.shape)
print('Valid images : ', valid_X.shape, valid_Y.shape)
print('Test  images : ', test_X.shape, test_Y.shape)

with h5py.File('data/train.h5', 'w') as dataset:
    dataset.create_dataset(data = train_X, name = 'X', shape = train_X.shape, dtype = train_X.dtype)
    dataset.create_dataset(data = train_Y, name = 'Y', shape = train_Y.shape, dtype = train_Y.dtype)

with h5py.File('data/valid.h5', 'w') as dataset:
    dataset.create_dataset(data = valid_X, name = 'X', shape = valid_X.shape, dtype = valid_X.dtype)
    dataset.create_dataset(data = valid_Y, name = 'Y', shape = valid_Y.shape, dtype = valid_Y.dtype)

with h5py.File('data/test.h5', 'w') as dataset:
    dataset.create_dataset(data = test_X, name = 'X', shape = test_X.shape, dtype = test_X.dtype)
    dataset.create_dataset(data = test_Y, name = 'Y', shape = test_Y.shape, dtype = test_Y.dtype)


