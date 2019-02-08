import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import h5py

imsize = 128
min_diff = 60
min_len = 32
scale = 1.0 / 255.0

def drawLines(images, paths):

	short_len, long_len = 0, 0
	while long_len - short_len <= min_diff:
		short_len, long_len = np.random.randint(min_len, imsize - 10, 2)

	lens = [short_len, long_len]

	for image_path, image, image_len in zip(paths, images, lens):
		start, end, ang = (-1, -1), (-1, -1), -1
		while any(cord < 0 or cord > imsize for cord in list(end)) is True:
			start = tuple(np.random.randint(0, imsize, 2))
			ang = np.random.randint(0, 360) * np.pi / 180
			end = int(start[0] + image_len * np.sin(ang)), int(start[1] + image_len * np.cos(ang))


		cv2.line(image, start, end, (255, 255, 255), 2)
		cv2.imwrite(image_path, image)

def draw(stage, num_images):


	for i in range(num_images):
		short_image = np.zeros((imsize, imsize))
		long_image = np.zeros_like(short_image)
		drawLines([short_image, long_image], ['data/{}/{}_short.jpg'.format(stage, i), 'data/{}/{}_long.jpg'.format(stage, i)])

def getHDF5():


	for stage in ['train', 'valid']:
		stage_images = []
		stage_labels = []
		image_ids = []
		num_images = len(os.listdir('data/{}'.format(stage))) // 2
		for i in range(num_images):
			short_image = cv2.imread('data/{}/{}_short.jpg'.format(stage, i), 0)
			long_image = cv2.imread('data/{}/{}_long.jpg'.format(stage, i), 0)
			if i % 2 == 0:
				stage_images.append(np.stack([short_image, long_image]))
				stage_labels.append(0)
			else:
				stage_images.append(np.stack([long_image, short_image]))
				stage_labels.append(1)
		if 'train' in stage:
			train_X = np.float32(stage_images) * scale
			train_Y = np.float32(stage_labels)
		else:
			valid_X = np.float32(stage_images) * scale
			valid_Y = np.float32(stage_labels)			

	np.random.seed(1729)
	imsize = train_X.shape[1]

	print('Train images : ', train_X.shape, train_Y.shape)
	print('Valid images : ', valid_X.shape, valid_Y.shape)

	indices = np.arange(train_X.shape[0])
	np.random.shuffle(indices)
	with h5py.File('data/train.h5', 'w') as dataset:
	    dataset.create_dataset(data = train_X[indices], name = 'X', shape = train_X.shape, dtype = train_X.dtype)
	    dataset.create_dataset(data = train_Y[indices], name = 'Y', shape = train_Y.shape, dtype = train_Y.dtype)

	indices = np.arange(valid_X.shape[0])
	np.random.shuffle(indices)
	with h5py.File('data/valid.h5', 'w') as dataset:
	    dataset.create_dataset(data = valid_X[indices], name = 'X', shape = valid_X.shape, dtype = valid_X.dtype)
	    dataset.create_dataset(data = valid_Y[indices], name = 'Y', shape = valid_Y.shape, dtype = valid_Y.dtype)


shutil.rmtree('data/train')
shutil.rmtree('data/valid')
os.mkdir('data/train')
os.mkdir('data/valid')
draw('train', 1000)
draw('valid', 100)
getHDF5()
