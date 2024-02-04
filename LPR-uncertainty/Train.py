from utils.datasets import Dataset
from utils.config import get_dicts_train
from architecture.model import BuildModel
from utils.data_generator import DataGenerator
from utils.trainer import Trainer

import tensorflow as tf
import keras 
from keras.losses import mean_absolute_error, categorical_crossentropy
from utils.metric import ssim, psnr,lp_val,lp_loss
from keras.metrics import categorical_accuracy
from keras import backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#define loss functions and metrics for the classification task depending on the dataset that is used
loss_cl = {"svhn":categorical_crossentropy,'mnist':categorical_crossentropy,"czech":lp_loss}
metric_cl = {"svhn":categorical_accuracy,'mnist':categorical_accuracy,"czech":lp_val}

data,parameter,design,store,training = get_dicts_train()

#create a dataset using the parsed arguments related to data
dataset = Dataset(data=data)
# preload the hdf5 files for faster access
dataset.load_files()

num_val_images = 64
num_train_images = 64

assert (training["n_batch"] <= num_train_images and training["n_batch"] <= num_val_images), \
    f"batch size smaller than number of training and validation images. " \
    f"Please change batch-size or in Train.py line 26 and 27" \
    f"\n num_val_images {num_val_images}" \
    f"\n num_train_images {num_train_images}" \
    f"\n batch-size {training['n_batch']}"

n_batch = training["n_batch"]

# load the data
X_lr_val,X_hr_val,l_val = dataset.load_data(mode='val',num=5000)

# Adjust brigtness of the validation data
X_lr_val = tf.image.random_brightness(X_lr_val,0.0,0.51)


#X_lr_train,X_hr_train,l_train =dataset.load_data(mode='train',num=8*25)

if parameter["batchEnsemble"]:
    num_images = X_lr_val.shape[0]
    rest = num_images % n_batch
    if rest != 0:
        cut =-rest
        X_lr_val = X_lr_val[:cut,]
        X_hr_val = X_hr_val[:cut, ]
        l_val = l_val[:cut, ]

    X_lr_val = np.split(X_lr_val,l_val.shape[0]//n_batch)
    X_hr_val = np.split(X_hr_val, l_val.shape[0] // n_batch)
    l_val = np.split(l_val, l_val.shape[0] // n_batch)

    for i in range(len(X_lr_val)):
        X_lr_val[i] = np.tile(X_lr_val[i], [ parameter["ensemble_size"], 1, 1, 1])
        X_hr_val[i] = np.tile(X_hr_val[i], [parameter["ensemble_size"], 1, 1, 1])
        l_val[i] = np.tile(l_val[i], [parameter["ensemble_size"], 1])
    # Additional batchsize parameter necessary
    X_lr_val = np.concatenate(X_lr_val)
    X_hr_val = np.concatenate(X_hr_val)
    l_val = np.concatenate(l_val)
    n_batch *= parameter["ensemble_size"]

    # Only necessary, if trainer.train is used
    #X_lr_train = np.tile(X_lr_train, [ parameter["ensemble_size"], 1, 1, 1])
    #X_hr_train = np.tile(X_hr_train, [parameter["ensemble_size"], 1, 1, 1])
    #l_train = np.tile(l_train, [parameter["ensemble_size"], 1])


generator = DataGenerator(dataset=dataset,training=training, parameter=parameter,model_type=design["model_type"])



# Build the model
builder = BuildModel(design=design,data=data,parameter=parameter)


model = builder.setup_model()

# setup training
trainer = Trainer(training=training,model=model,store=store)

# choose loss functions
loss_weights = [ K.variable(value=parameter["w_sr"], name='weight_sr'),
                 K.variable(value=parameter["w_cl"], name='weight_cl')]

loss = {'sr': mean_absolute_error, 'cl': loss_cl[data["dataset"]]}
metric = {'sr': ssim, 'cl': metric_cl[data["dataset"]]}

# set callbacks:
trainer.set_callbacks(["reduceLR","store","tb"])

# compile the model
trainer.compile_model(loss=loss,loss_weights=loss_weights,metric=metric)
# train the model

#trainer.train(X_lr_train, X_hr_train,l_train,X_lr_val, X_hr_val,l_val,n_batch=n_batch)
trainer.train_generator(generator=generator,X_lr_val=X_lr_val, X_hr_val=X_hr_val,l_val=l_val,n_batch=n_batch)
