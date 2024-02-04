import os
import random

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
from skimage.filters import gaussian
from skimage.transform import rescale
from random import sample
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter


class Dataset:
    def __init__(self,data):
        self.in_shape = data["in_shape"]
        self.n_channels = data["n_channels"]
        self.n_classes = data["n_classes"]
        self.scale = data["scale"]
        self.dataset = data["dataset"]
        self.noise = data["noise"]
        self.noiseType = data["noiseType"]
        self.n_low = data["noise_low"]
        self.n_high = data["noise_high"]
        self.blur = data["blur"]
        self.blurType = data["blurType"]
        self.b_strength = data["blur_strength"]
        self.b_kernel = data["blur_kernel"]
        self.local_dir = data["local_dir"]
        self.data_dir = os.path.join(self.local_dir, self.dataset)
        self.out_shape = data["out_shape"]
        self.ending = "hdf5"
        if self.dataset == "czech":
            self.ending = "h5"

        if self.noise:
            if self.noiseType == "gaussian":
                self.noiser = self.gaussian
            elif self.noiseType == "sp":
                self.noiser = self.sp
            elif self.noiseType == "speckle":
                self.noiser = self.speckle
        else:
            self.noiser = self.empty

        if self.blur:
            if self.blurType == "horizontal":
                self.blurrer = self.horizontal
            elif self.blurType == "vertical":
                self.blurrer = self.vertical
            elif self.blurType == "defocus":
                self.blurrer = self.defocus
        else:
            self.blurrer = self.empty

        if not os.path.isfile(os.path.join(self.data_dir, 'train/idx_list.txt')):
            print("Please store the data using dataset.store_data()")



    def load_files(self):

        self.IDs = {}
        f = open(os.path.join(self.data_dir, 'train/idx_list.txt'), 'r')
        self.IDs["train"] = (f.read().splitlines())
        f.close()
        f = open(os.path.join(self.data_dir, 'val/idx_list.txt'), 'r')
        self.IDs["val"] = (f.read().splitlines())
        f.close()
        f =open(os.path.join(self.data_dir, 'test/idx_list.txt'), 'r')
        self.IDs["test"] = (f.read().splitlines())
        f.close()

        self.files = {}
        self.files["train"] = h5py.File(os.path.join(self.data_dir,"train/train.{0}".format(self.ending)), "r")
        self.files["val"] = h5py.File(os.path.join(self.data_dir, "val/val.{0}".format(self.ending)), "r")
        self.files["test"] = h5py.File(os.path.join(self.data_dir, "test/test.{0}".format(self.ending)), "r")

    def load_data_generator_cl(self,IDs_batch,batchsize, batchEnsemble, ensemble_size):
        X_lr_batch = np.empty((batchsize, self.in_shape[0], self.in_shape[1], self.n_channels))
        l_batch = np.empty((batchsize, self.n_classes), dtype=int)
        for i,ID in enumerate(IDs_batch):
            tmp = self.blurrer((self.files['train']['images_lr{0}'.format(self.scale)])[int(ID)])
            X_lr_batch[i,] = self.noiser(tmp)
            l_batch[i,] = (self.files['train']['labels'])[int(ID)]

        if batchEnsemble:
            X_lr_batch = np.tile(X_lr_batch, [ensemble_size, 1, 1, 1])
            l_batch = np.tile(l_batch, [ensemble_size, 1])

        return X_lr_batch,l_batch

    def load_data_generator_sr(self,IDs_batch,batchsize, batchEnsemble, ensemble_size):
        X_hr_batch = np.empty((batchsize, self.out_shape[0], self.out_shape[1], self.n_channels))
        X_lr_batch = np.empty((batchsize, self.in_shape[0], self.in_shape[1], self.n_channels))

        for i,ID in enumerate(IDs_batch):
            X_hr_batch[i,] = (self.files['train']['images_hr'])[int(ID)]
            tmp = self.blurrer((self.files['train']['images_lr{0}'.format(self.scale)])[int(ID)])
            X_lr_batch[i, ] = self.noiser(tmp)

        if batchEnsemble:
            X_lr_batch = np.tile(X_lr_batch, [ensemble_size, 1, 1, 1])
            X_hr_batch = np.tile(X_hr_batch, [ensemble_size, 1, 1, 1])

        return X_lr_batch,X_hr_batch

    def load_data_generator_sr2(self,IDs_batch,batchsize, batchEnsemble, ensemble_size):
        X_hr_batch = np.empty((batchsize, self.out_shape[0], self.out_shape[1], self.n_channels),dtype=np.float32)
        X_lr_batch = np.empty((batchsize, self.in_shape[0], self.in_shape[1], self.n_channels),dtype=np.float32)
        l_batch = np.empty((batchsize, self.n_classes), dtype=int)

        for i,ID in enumerate(IDs_batch):
            X_hr_batch[i,] = (self.files['train']['images_hr'])[int(ID)]
            tmp = self.blurrer((self.files['train']['images_lr{0}'.format(self.scale)])[int(ID)])
            X_lr_batch[i, ] = self.noiser(tmp)
            l_batch[i,] = (self.files['train']['labels'])[int(ID)]

        if batchEnsemble:
            X_lr_batch = np.tile(X_lr_batch, [ensemble_size, 1, 1, 1])
            X_hr_batch = np.tile(X_hr_batch, [ensemble_size, 1, 1, 1])
            l_batch = np.tile(l_batch, [ensemble_size, 1])

        return X_lr_batch,[X_hr_batch,l_batch]


    # possible modes are: train, val, test
    # num specifies the number of samples that are used , if set to a value < 0, the number of images available is used
    def load_data(self,mode,num,sampling=False):
        # check whether the file exists


        if not os.path.isfile(os.path.join(self.data_dir,"{0}/{0}.{1}".format(mode,self.ending))):
            if self.dataset == "MNIST":
                self.store_data()
            else:
                raise Exception("Data not found! Please download {0}".format(self.dataset))
        #load file IDs

        if num <= len(self.IDs[mode]) and num > 0:
            if sampling:
                self.IDs[mode] = sample(self.IDs[mode],num)
            else:
                self.IDs[mode] = self.IDs[mode][0:num]
        else:
            num = len(self.IDs[mode])


        # load data
        fileData = self.files[mode]
        X_hr = np.empty((num, self.out_shape[0], self.out_shape[1], self.n_channels),dtype=np.float32)
        X_lr = np.empty((num, self.in_shape[0], self.in_shape[1], self.n_channels),dtype=np.float32)
        l = np.empty((num, self.n_classes), dtype=int)
        for i,ID in enumerate(self.IDs[mode]):
            X_hr[i, ] = (fileData['images_hr'])[int(ID)]

            tmp = (fileData['images_lr{0}'.format(self.scale)])[int(ID)]
            tmp = self.blurrer(tmp)
            X_lr[i, ] = self.noiser(tmp)
            l[i,] = (fileData['labels'])[int(ID)]


        return X_lr,X_hr,l

    def gaussian(self,X_lr):
        gaussian = np.random.normal(0, random.uniform(self.n_low, self.n_high),
                                            (self.in_shape[0], self.in_shape[1], self.n_channels))
        X_lr = X_lr + gaussian
        X_lr[X_lr > 1] = 1
        X_lr[X_lr < 0] = 0

        return X_lr


    def speckle(self,X_lr):
        gaussian =  np.random.normal(0, random.uniform(self.n_low, self.n_high),
                                            (self.in_shape[0], self.in_shape[1], self.n_channels))

        X_lr = X_lr + X_lr * gaussian
        X_lr[X_lr > 1] = 1
        X_lr[X_lr < 0] = 0

        return X_lr

    def sp(self,X_lr):

        p = random.uniform(self.n_low, self.n_high)
        flipped= np.random.choice([True, False], size=(self.in_shape[0], self.in_shape[1], self.n_channels),
                                           p=[p, 1 - p])

        salted = np.random.choice([True, False], size=( self.in_shape[0], self.in_shape[1], self.n_channels),
                                  p=[0.5, 0.5])
        peppered = ~salted
        X_lr[flipped & salted] = 1.0
        X_lr[flipped & peppered] = 0.0

        return X_lr

    def horizontal(self,X_lr):
        kernel_motion_blur = np.zeros((self.b_kernel, self.b_kernel))
        kernel_motion_blur[int((self.b_kernel - 1) / 2), :] = np.ones(self.b_kernel)
        kernel_motion_blur = kernel_motion_blur / self.b_kernel
        X_lr[:,:,0] = convolve2d(X_lr[:,:,0], kernel_motion_blur, 'same')

        return X_lr

    def vertical(self,X_lr):
        kernel_motion_blur = np.zeros((self.b_kernel, self.b_kernel))
        kernel_motion_blur[int((self.b_kernel - 1) / 2), :] = np.ones(self.b_kernel)
        kernel_motion_blur = (kernel_motion_blur / self.b_kernel).T
        X_lr[:,:,0] = convolve2d(X_lr[:,:,0], kernel_motion_blur, 'same')

        return X_lr

    def defocus(self,X_lr):
        X_lr = gaussian_filter(X_lr, sigma=self.b_strength)
        return X_lr


    def empty(self,X_lr):
        return X_lr

    def create_down(self,input, scale):
        # first blur the data
        # multi channel is always True, because grayscale images are stored as (32,32,1)
        input = gaussian(input.astype(np.float64), sigma=scale / 2, multichannel=True)
        data_down = rescale(input, 1 / scale, multichannel=True)
        return data_down.astype(np.float16)



    def store_data(self):
        train_data_hr, train_label, val_data_hr, val_label, test_data_hr, test_label = self.get_data()
        test_data_hr = test_data_hr.astype(np.float16)
        val_data_hr = val_data_hr.astype(np.float16)
        train_data_hr = train_data_hr.astype(np.float16)

        if np.max(test_data_hr[0,] > 1):
            test_data_hr = test_data_hr / 255
            val_data_hr = val_data_hr / 255
            train_data_hr = train_data_hr / 255

        # store the test data
        if not os.path.exists(os.path.join(self.data_dir, 'test')):
            os.makedirs(os.path.join(self.data_dir, 'test'), exist_ok=True)
        fileTest = open(os.path.join(os.path.join(self.data_dir, 'test'), 'idx_list.txt'), 'w')
        sTest = test_data_hr.shape
        down2 = np.zeros((sTest[0], int(sTest[1] / 2), int(sTest[2] / 2), sTest[3]))
        down4 = np.zeros((sTest[0], int(sTest[1] / 4), int(sTest[2] / 4), sTest[3]))
        for idx_test in range(test_data_hr.shape[0]):
            fileTest.write('{0:06d}\n'.format(idx_test))
            down2[idx_test,] = self.create_down(test_data_hr[idx_test,], 2)
            down4[idx_test,] = self.create_down(test_data_hr[idx_test,], 4)
        fileTest.close()

        with h5py.File(os.path.join(os.path.join(self.data_dir, 'test'), 'test.hdf5'), "w") as g:
            g.create_dataset('images_hr', data=test_data_hr)
            g.create_dataset('images_lr2', data=down2)
            g.create_dataset('images_lr4', data=down4)
            g.create_dataset('labels', data=test_label)

        del test_data_hr
        del test_label
        del down2
        del down4

        # store the val data
        if not os.path.exists(os.path.join(self.data_dir, 'val')):
            os.makedirs(os.path.join(self.data_dir, 'val'), exist_ok=True)
        fileVal = open(os.path.join(os.path.join(self.data_dir, 'val'), 'idx_list.txt'), 'w')
        sVal = val_data_hr.shape
        down2 = np.zeros((sVal[0], int(sVal[1] / 2), int(sVal[2] / 2), sVal[3]))
        down4 = np.zeros((sVal[0], int(sVal[1] / 4), int(sVal[2] / 4), sVal[3]))
        for idx_val in range(val_data_hr.shape[0]):
            fileVal.write('{0:06d}\n'.format(idx_val))
            down2[idx_val,] = self.create_down(val_data_hr[idx_val,], 2)
            down4[idx_val,] = self.create_down(val_data_hr[idx_val,], 4)
        fileVal.close()

        with h5py.File(os.path.join(os.path.join(self.data_dir, 'val'), 'val.hdf5'), "w") as g:
            g.create_dataset('images_hr', data=val_data_hr)
            g.create_dataset('images_lr2', data=down2)
            g.create_dataset('images_lr4', data=down4)
            g.create_dataset('labels', data=val_label)

        del val_data_hr
        del val_label
        del down2
        del down4

        # store the train data
        if not os.path.exists(os.path.join(self.data_dir, 'train')):
            os.makedirs(os.path.join(self.data_dir, 'train'), exist_ok=True)
        fileTrain = open(os.path.join(os.path.join(self.data_dir, 'train'), 'idx_list.txt'), 'w')
        sTrain = train_data_hr.shape
        down2 = np.zeros((sTrain[0], int(sTrain[1] / 2), int(sTrain[2] / 2), sTrain[3]))
        down4 = np.zeros((sTrain[0], int(sTrain[1] / 4), int(sTrain[2] / 4), sTrain[3]))
        for idx_train in range(train_data_hr.shape[0]):
            fileTrain.write('{0:06d}\n'.format(idx_train))
            down2[idx_train,] = self.create_down(train_data_hr[idx_train,], 2)
            down4[idx_train,] = self.create_down(train_data_hr[idx_train,], 4)
        fileTrain.close()

        with h5py.File(os.path.join(os.path.join(self.data_dir, 'train'), 'train.hdf5'), "w") as g:
            g.create_dataset('images_hr', data=train_data_hr)
            g.create_dataset('images_lr2', data=down2)
            g.create_dataset('images_lr4', data=down4)
            g.create_dataset('labels', data=train_label)

        del train_data_hr
        del train_label
        del down2
        del down4


    def get_data(self):
        if self.dataset == "mnist":
            # download MNIST
            (train_data_hr, train_label), (test_data_hr, test_label) = tf.keras.datasets.mnist.load_data()
            # get size of the data
            num_train_all,w,h = train_data_hr.shape[0],train_data_hr.shape[1], train_data_hr.shape[2]
            num_test = test_data_hr.shape[0]

            # add additional dimension for the network
            train_data_hr = train_data_hr.reshape((num_train_all,w,h, 1))
            test_data_hr = test_data_hr.reshape((num_test,w,h, 1))

            num_val = int(0.2*num_train_all)
            val_data_hr = train_data_hr[0:num_val,]
            val_label = train_label[0:num_val,]

            train_data_hr = train_data_hr[num_val:num_train_all,]
            train_label = train_label[num_val:num_train_all,]

            return train_data_hr, train_label, val_data_hr, val_label, test_data_hr, test_label

        elif self.dataset == "svhn":
            train = sio.loadmat(self.data_dir + '/train_32x32.mat')
            train_data_hr_tmp, l_train = train['X'], train['y']
            trainextra = sio.loadmat(self.data_dir + '/extra_32x32.mat')
            trainextra_data_hr_tmp, trainextra_label = trainextra['X'], trainextra['y']
            test = sio.loadmat(self.data_dir + '/test_32x32.mat')
            test_data_hr_tmp, l = test['X'], test['y']

            ## change order
            train_data_hr = np.moveaxis(train_data_hr_tmp, -1, 0)
            trainextra_data_hr = np.moveaxis(trainextra_data_hr_tmp, -1, 0)
            test_data_hr = np.moveaxis(test_data_hr_tmp, -1, 0)


            # only use 70000 additional images because of performance
            trainextra_data_hr = trainextra_data_hr[0:70000,]
            trainextra_label = trainextra_label[0:70000,]

            # split into training and validation
            num_train_all = train_data_hr.shape[0]
            num_val = int(0.2*num_train_all)

            val_data_hr = train_data_hr[0:num_val,]
            l_val = l_train[0:num_val, ]

            ## append extra data
            train_data_hr = np.append(train_data_hr[num_val:num_train_all,], trainextra_data_hr, axis=0)
            l_train = np.append(l_train[num_val:num_train_all,], trainextra_label, axis=0)

            # adapt labels
            train_label = np.zeros((l_train.shape[0], 10))
            for i in range(l_train.shape[0]):
                pos = np.mod(l_train[i,], 10)
                train_label[i, pos] = 1

            val_label = np.zeros((l_val.shape[0], 10))
            for i in range(l_val.shape[0]):
                pos = np.mod(l_val[i,], 10)
                val_label[i, pos] = 1

            test_label = np.zeros((l.shape[0], 10))
            for i in range(l.shape[0]):
                pos = np.mod(l[i,], 10)
                test_label[i, pos] = 1
            return train_data_hr, train_label, val_data_hr, val_label, test_data_hr, test_label


