import cv2
import struct
import os
import numpy as np
import h5py

def writeData():

	paths = ['data/denom/train', 'data/denom/valid']
	images = [[], []]
	labels = [[], []]
	for i, path in enumerate(paths):
	    for item in os.listdir(path):
	        image_path = os.path.join(path, item)
	        images[i].append(cv2.imread(image_path, 0))
	        labels[i].append(int(float(item.split('_')[0])))

	train_X, valid_X = np.float32(images[0]), np.float32(images[1])
	train_Y, valid_Y = np.float32(labels[0]), np.float32(labels[1]) 

	np.random.seed(1729)
	imsize = train_X.shape[1]

	mean, sigma = 0.0, 255.0
	train_X, train_Y = ((np.float32(train_X) - mean) / sigma).reshape(-1, 1, imsize, imsize), np.float32(train_Y)
	valid_X, valid_Y = ((np.float32(valid_X) - mean) / sigma).reshape(-1, 1, imsize, imsize), np.float32(valid_Y)

	print('Train images : ', train_X.shape, train_Y.shape)
	print('Valid images : ', valid_X.shape, valid_Y.shape)

	indices = np.arange(train_X.shape[0])
	np.random.shuffle(indices)
	with h5py.File('data/denom/train.h5', 'w') as dataset:
	    dataset.create_dataset(data = train_X[indices], name = 'X', shape = train_X.shape, dtype = train_X.dtype)
	    dataset.create_dataset(data = train_Y[indices], name = 'Y', shape = train_Y.shape, dtype = train_Y.dtype)

	indices = np.arange(valid_X.shape[0])
	np.random.shuffle(indices)
	with h5py.File('data/denom/valid.h5', 'w') as dataset:
	    dataset.create_dataset(data = valid_X[indices], name = 'X', shape = valid_X.shape, dtype = valid_X.dtype)
	    dataset.create_dataset(data = valid_Y[indices], name = 'Y', shape = valid_Y.shape, dtype = valid_Y.dtype)


writeData()