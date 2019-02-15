import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import h5py

imsize = 128
scale = 1.0 / 255.0

## Line lengths, distance between lines, line thickness

def drawLine(image, line_len, angle, thickness):
    start, end, ang = (-1, -1), (-1, -1), -1
    while any(cord < 0 or cord > imsize for cord in list(end)) is True:
        start = tuple(np.random.randint(0, imsize, 2))
        end = int(start[0] + line_len * np.sin(angle)), int(start[1] + line_len * np.cos(angle))
    cv2.line(image, start, end, (255, 255, 255), thickness)

def drawLines(image, is_parallel, image_path, min_len, max_len, angle_diff, num_lines = 2):

    base_angle = np.random.randint(0, 180) * np.pi / 180
    line_len = np.random.randint(min_len, max_len)
    thickness = np.random.randint(1, 4)
    drawLine(image, line_len, base_angle, thickness)

    for i in range(num_lines - 1):
        if is_parallel is False:
            angle = base_angle + 100
            while np.abs(angle - base_angle) > angle_diff:
                angle = np.random.randint(0, 180) * np.pi / 180
        else:
            angle = base_angle
        line_len = np.random.randint(min_len, max_len)
        thickness = np.random.randint(1, 4)
        drawLine(image, line_len, angle, thickness)

    cv2.imwrite(image_path, image)

def draw(stage, num_images, seed, min_len, max_len, angle_diff):

    np.random.seed(seed)

    for i in range(num_images):
        image = np.zeros((imsize, imsize))
        is_parallel = True if i % 2 == 0 else False
        drawLines(image, is_parallel, 'data/{}/{}.jpg'.format(stage, i), min_len, max_len, angle_diff)

def getHDF5():

    for stage in ['train', 'valid', 'test']:
        stage_images = []
        stage_labels = []
        image_ids = []
        num_images = len(os.listdir('data/{}'.format(stage)))
        for i in range(num_images):
            image = cv2.imread('data/{}/{}.jpg'.format(stage, i), 0)
            if i % 2 == 0:
                stage_images.append(image)
                stage_labels.append(0)
            else:
                stage_images.append(image)
                stage_labels.append(1)
        if 'train' in stage:
            train_X = np.float32(stage_images) * scale
            train_Y = np.float32(stage_labels)
        elif 'valid' in stage:
            valid_X = np.float32(stage_images) * scale
            valid_Y = np.float32(stage_labels)			
        else:
            test_X = np.float32(stage_images) * scale
            test_Y = np.float32(stage_labels)

    np.random.seed(1729)
    imsize = train_X.shape[1]

    print('Train images : ', train_X.shape, train_Y.shape)
    print('Valid images : ', valid_X.shape, valid_Y.shape)
    print('Test images : ', test_X.shape, test_Y.shape)

    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    with h5py.File('/scratch/scratch2/karthikt/data/train.h5', 'w') as dataset:
        dataset.create_dataset(data = train_X[indices], name = 'X', shape = train_X.shape, dtype = train_X.dtype)
        dataset.create_dataset(data = train_Y[indices], name = 'Y', shape = train_Y.shape, dtype = train_Y.dtype)

    indices = np.arange(valid_X.shape[0])
    np.random.shuffle(indices)
    with h5py.File('/scratch/scratch2/karthikt/data/valid.h5', 'w') as dataset:
        dataset.create_dataset(data = valid_X[indices], name = 'X', shape = valid_X.shape, dtype = valid_X.dtype)
        dataset.create_dataset(data = valid_Y[indices], name = 'Y', shape = valid_Y.shape, dtype = valid_Y.dtype)

    indices = np.arange(test_X.shape[0])
    np.random.shuffle(indices)
    with h5py.File('/scratch/scratch2/karthikt/data/test.h5', 'w') as dataset:
        dataset.create_dataset(data = test_X[indices], name = 'X', shape = test_X.shape, dtype = test_X.dtype)
        dataset.create_dataset(data = test_Y[indices], name = 'Y', shape = test_Y.shape, dtype = test_Y.dtype)

if os.path.isdir('data/train'):
    shutil.rmtree('data/train')
if os.path.isdir('data/valid'):
    shutil.rmtree('data/valid')
if os.path.isdir('data/test'):
    shutil.rmtree('data/test')
os.mkdir('data/train')
os.mkdir('data/valid')
os.mkdir('data/test')
draw('train', 100000, 1001, 30, 40, 30)
draw('valid', 1000, 1729, 20, 50, 20)
draw('test', 1000, 1123, 10, 60, 10)
getHDF5()
