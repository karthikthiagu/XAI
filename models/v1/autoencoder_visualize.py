import os
import shutil
import numpy as np
import cv2
import h5py

from sklearn.cluster import SpectralClustering

from argparser import parseArguments

def loadData(path, num_maps):
    data = h5py.File(path, 'r')
    imsize = 128
    size = int(num_maps * 16 * 16)
    return data['rX'][:].reshape((1000, 128, 128)), data['X'][:].reshape((-1, size)), data['Y'][:]


################# Argument parsing ########################
###########################################################
args = parseArguments()
feats_file = args.test_file
num_maps = args.num_maps
plot_folder = args.plot_folder

################## Load data #########################
######################################################
rX, X, Y = loadData(feats_file, num_maps)

################# Cluster ###########################
#####################################################
cluster = SpectralClustering(n_clusters = 2, affinity = 'nearest_neighbors', n_neighbors = 50, assign_labels = 'discretize', random_state = 0)
cluster.fit(X)
pred_Y = np.int16(cluster.labels_)
Y = np.int16(Y)
print((Y == pred_Y).mean())

######## Plot folder setup
if os.path.isdir(plot_folder):
    shutil.rmtree(plot_folder)
os.mkdir(plot_folder)

############## Reconstruction ################
##############################################
for i in range(rX.shape[0])[ : 100]:
    plot_path = '{}/{}.jpg'.format(plot_folder, i)
    print(plot_path)
    cv2.imwrite(plot_path, rX[i] / np.max(rX[i]) * 255.0)

