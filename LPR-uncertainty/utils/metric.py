import tensorflow as tf
import keras
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

def psnr(hr, sr):
    return tf.image.psnr(hr, sr,1)

def ssim(hr,sr):
    return tf.image.ssim(hr,sr,1)


def lp_loss(y_true,y_pred):

    l = 0
    for i in range(7):
        l+= tf.reduce_mean(categorical_crossentropy(y_true=y_true[:,i*37:(i+1)*37], y_pred=y_pred[:,i*37:(i+1)*37]))
    l /= 7
    return l

def lp_val(y_true,y_pred):

    l = 0
    for i in range(7):
        l+= tf.reduce_mean(categorical_accuracy(y_true=y_true[:,i*37:(i+1)*37], y_pred=y_pred[:,i*37:(i+1)*37]))
    l /= 7
    return l


def lp_loss_ccpd(y_true,y_pred):

    l = 0
    for i in range(7):
        l+= tf.reduce_mean(categorical_crossentropy(y_true=y_true[:,i*68:(i+1)*68], y_pred=y_pred[:,i*68:(i+1)*68]))
    l /= 7
    return l

def lp_val_ccpd(y_true,y_pred):

    l = 0
    for i in range(7):
        l+= tf.reduce_mean(categorical_accuracy(y_true=y_true[:,i*68:(i+1)*68], y_pred=y_pred[:,i*68:(i+1)*68]))
    l /= 7
    return l