import functools
import tensorflow as tf
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Lambda, GlobalAveragePooling2D, \
    Add, MaxPooling2D,Flatten,Concatenate,Dropout, ReLU, BatchNormalization
from keras.initializers import RandomSign
from keras.layers.Conv2D import Conv2DBatchEnsemble
from keras.layers.normalization import  ensemble_batchnorm
from keras.layers.dense import DenseBatchEnsemble


# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5



def make_random_sign_initializer(random_sign_init):
  if random_sign_init > 0:
    initializer = RandomSign(random_sign_init)
  else:
    initializer = tf.keras.initializers.RandomNormal(mean=1.0,
                                                     stddev=-random_sign_init)
  return initializer


def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.compat.v1.depth_to_space(x, scale), **kwargs)

## Basic residual blocks

def block_res(inputs,num_identity, n_filter=16, kernel_size=3,reg_strength=0.001):
    x = inputs
    for i in range(num_identity):
        x = res_identity(x,n_filter=n_filter,kernel_size=kernel_size,reg_strength=reg_strength,name='sr_i{0}'.format(i))
    return x




def res_identity(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test'):
    x = inputs
    x = BatchNormalization(name='{0}_bn1'.format(name))(x)
    x = ReLU()(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='{0}_bn2'.format(name))(x)
    x = ReLU()(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv2'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    return Add(name='{0}_add'.format(name))([x, inputs])


def res_down(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test'):

    x = inputs
    x = BatchNormalization(name='{0}_large_bn1'.format(name))(x)
    x = ReLU()(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=2,
                  padding='same', name='{0}_large_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='{0}_large_bn2'.format(name))(x)
    x = ReLU()(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_large_conv2'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)


    y = inputs
    y = BatchNormalization(name='{0}_small_bn1'.format(name))(y)
    y = ReLU()(y)
    y = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=2,
                  padding='same',name='{0}_small_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(y)
    return Add(name='{0}_add'.format(name))([x, y])

## All possible common layers

def common_fsrcnn(inputs,reg_strength=0.001,dropout=False,dropout_rate=0.5,
                     batch_ensemble=False, ensemble_size=5):
    x = inputs

    if dropout:
        x = Dropout(rate=dropout_rate)(x, training=True)
        x = Conv2D(56,kernel_size=5,strides=1,padding='same',name='last_common',
                   kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
        x = ReLU()(x)
        x = Dropout(rate=dropout_rate)(x, training=True)

        return x

    if batch_ensemble:
        x = Conv2DBatchEnsemble(
            56, kernel_size=5, use_bias=False, padding='same', strides=1,
            kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer='he_normal',
            alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
            name='last_common', ensemble_size=ensemble_size)(x)
        x = ensemble_batchnorm(x, ensemble_size=ensemble_size, name='common_bn', use_tpu=False)
        x = ReLU()(x)

        return x

    x = Conv2D(56,kernel_size=5,strides=1,padding='same',name='last_common',
                   kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='common_bn')(x)
    x = ReLU()(x)

    return x

def common_block_res(inputs,split, n_filter=16, kernel_size=3,reg_strength=0.001,dropout=False,dropout_rate=0.5,
                     batch_ensemble=False, ensemble_size=5):
    x = inputs

    random_sign_init = -0.5

    if (split == 0):
        if dropout:
            x = Dropout(rate=dropout_rate)(x, training=True)
        return x


    if batch_ensemble:
        x = Conv2DBatchEnsemble(
            n_filter, kernel_size=3, use_bias=False, padding='same', strides=1,
            kernel_regularizer=regularizers.l2(reg_strength),
            alpha_initializer=make_random_sign_initializer(random_sign_init),
            gamma_initializer=make_random_sign_initializer(random_sign_init),
            name='common_res_first', ensemble_size=ensemble_size)(x)
    else:
        factor = 1
        if dropout: factor = (1 / (1 - dropout_rate)) ** (0.5)

        x = Conv2D(int(n_filter*factor), 3, padding='same', strides=1, kernel_initializer=he_normal(), name='common_res_first'
                   ,kernel_regularizer=regularizers.l2(reg_strength))(x)

    if batch_ensemble:
        for i in range(split):
            x = res_identity_be(x, n_filter=n_filter,
                                kernel_size=kernel_size, reg_strength=reg_strength,
                                name='common_res_i{0}'.format(i),
                                ensemble_size=ensemble_size)
        return x

    if dropout:
        for i in range(split):
            x = res_identity_dropout(x, n_filter=n_filter,
                                     kernel_size=kernel_size, reg_strength=reg_strength,
                                     name='common_res_i{0}'.format(i),
                                     dropout_rate=dropout_rate)
        return x

    for i in range(split):
        x = res_identity(x, n_filter=n_filter,
                            kernel_size=kernel_size, reg_strength=reg_strength,
                            name='common_res_i{0}'.format(i))

    return x

def common_block_conv(inputs,split=4,n_filter=16,kernel_size=3,reg_strength=0.001,dropout=False,dropout_rate=0.5,
                     batch_ensemble=False, ensemble_size=5):
    random_sign_init = -0.5
    if (split == 0): return inputs

    x = inputs

    if dropout:
        factor = (1 / (1 - dropout_rate)) ** (0.5)
        x = Dropout(rate=dropout_rate)(x, training=True)
        for i in range(split):
            x = Conv2D(n_filter*factor,kernel_size=5,strides=1,padding='same',name=f'common_conv_conv{i}',
                       kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
            x = BatchNormalization(name=f'common_conv_bn{i}')(x)
            x = ReLU()(x)
            x = Dropout(rate=dropout_rate)(x, training=True)

        return x

    if batch_ensemble:
        for i in range(split):
            x = Conv2DBatchEnsemble(
                n_filter, kernel_size=5, use_bias=False, padding='same', strides=1,
                kernel_regularizer=regularizers.l2(reg_strength),
                alpha_initializer=make_random_sign_initializer(random_sign_init),
                gamma_initializer=make_random_sign_initializer(random_sign_init),
                name=f'common_conv_conv{i}', ensemble_size=ensemble_size)(x)
            x = ensemble_batchnorm(x, ensemble_size=ensemble_size, name=f'common_conv_bn{i}', use_tpu=False)
            x = ReLU()(x)

        return x

    for i in range(split):
        x = Conv2D(n_filter, kernel_size=5, strides=1, padding='same',
                   name=f'common_conv_conv{i}',
                   kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength))(x)
        x = BatchNormalization(name=f'common_conv_bn{i}')(x)
        x = ReLU()(x)


    return x


## Super-Resolution Networks

def fsrcnn_red(inputs, scale=4, n_channels=3,reg_strength=0.001):
    x = inputs

    ## Conv. Layer 2: Shrinking
    x = Conv2D(12, kernel_size=1, padding='same',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_2_conv')(x )
    x = BatchNormalization(name='sr_2_bn')(x)
    x = ReLU()(x)

    ## Conv. Layers 3–6: Mapping
    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_3_conv')(x )
    x = BatchNormalization(name='sr_3_bn')(x)
    x = ReLU()(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_4_conv')(x )
    x = BatchNormalization(name='sr_4_bn')(x)
    x = ReLU()(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_5_conv')(x )
    x = BatchNormalization(name='sr_5_bn')(x)
    x = ReLU()(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_6_conv')(x )
    x = BatchNormalization(name='sr_6_bn')(x)
    x = ReLU()(x)

    ##Conv.Layer  7: Expanding
    x = Conv2D(56, kernel_size=1, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_7_conv')(x )
    x = BatchNormalization(name='sr_7_bn')(x)
    x = ReLU()(x)

    ##DeConv Layer 8: Deconvolution

    x = Conv2D(3 * scale ** 2, 9, padding='same', name='sr_8_conv_up{0}'.format(scale),
               kernel_initializer=he_normal())(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2D(1, 1, padding='same', name='sr', kernel_initializer=he_normal())(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1




def fsrcnn(inputs, scale=4, n_channels=3,reg_strength=0.001):
    x = inputs
    ## Conv. Layer 1: feature extraction layer 1
    x = Conv2D(56, kernel_size=5,padding='same',
                  kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_1_conv')(x)
    x = BatchNormalization(name='sr_1_bn')(x)
    x = ReLU()(x)

    ## Conv. Layer 2: Shrinking
    x = Conv2D(12, kernel_size=1, padding='same',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_2_conv')(x )
    x = BatchNormalization(name='sr_2_bn')(x)
    x = ReLU()(x)

    ## Conv. Layers 3–6: Mapping
    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_3_conv')(x )
    x = BatchNormalization(name='sr_3_bn')(x)
    x = ReLU()(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_4_conv')(x )
    x = BatchNormalization(name='sr_4_bn')(x)
    x = ReLU()(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_5_conv')(x )
    x = BatchNormalization(name='sr_5_bn')(x)
    x = ReLU()(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_6_conv')(x )
    x = BatchNormalization(name='sr_6_bn')(x)
    x = ReLU()(x)

    ##Conv.Layer  7: Expanding
    x = Conv2D(56, kernel_size=1, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_7_conv')(x )
    x = BatchNormalization(name='sr_7_bn')(x)
    x = ReLU()(x)

    ##DeConv Layer 8: Deconvolution

    x = Conv2D(3 * scale ** 2, 9, padding='same', name='sr_8_conv_up{0}'.format(scale),
               kernel_initializer=he_normal())(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2D(1, 1, padding='same', name='sr', kernel_initializer=he_normal())(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1


def wdsr(inputs, scale=4,n_res_blocks=16, n_channels=3,n_filter=32,reg_strength=0.001):
    x = inputs
    x = Conv2D(n_filter, 3, padding='same', name='sr_1_conv', kernel_initializer=he_normal())(x)
    ## res blocks main branch
    m = block_res(x, num_identity=n_res_blocks, n_filter=n_filter, kernel_size=3, reg_strength=reg_strength)
    ## Deconvolution
    m = Conv2D(3 * scale ** 2, 3, padding='same', name='sr_main_last_conv',kernel_initializer=he_normal())(m)
    m = BatchNormalization(name='sr_main_last_bn')(m)
    m = ReLU()(m)
    m = SubpixelConv2D(scale)(m)
    if n_channels == 1:
        m1 = Conv2D(1, 1, padding='same', name='sr_main_1dconv',kernel_initializer=he_normal())(m)
    else:
        m1 = m

    # skip branch
    s = Conv2D(3 * scale ** 2, 5, padding='same', name='sr_skip_last_conv',
               kernel_initializer=he_normal())(x)

    s = BatchNormalization(name='sr_skip_last_bn')(s)
    s = ReLU()(s)
    s = SubpixelConv2D(scale)(s)
    if n_channels == 1:
        s1 = Conv2D(1, 1, padding='same', name='sr_skip_1dconv',kernel_initializer=he_normal())(s)
    else:
        s1 = s

    return Add(name='sr')([m1, s1])

## Classification networks
def resnet_mine(inputs, n_classes=10, reg_strength=0.001):

    x = inputs
    x = Conv2D(64,kernel_size=7,strides=1,padding='same',name='cl_1_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_1_bn')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding="same", name='cl_max')(x)
    x = ReLU()(x)
    x = res_identity(x, 64, 3,reg_strength,name='cl_i64_1')
    x = res_identity(x, 64, 3, reg_strength, name='cl_i64_2')
    x = res_down(x, 128, 3,reg_strength,name='cl_d128')
    x = res_identity(x, 128, 3, reg_strength, name='cl_i128_1')
    x = res_down(x, 256, 3,reg_strength,name='cl_d256')
    x = res_identity(x, 256, 3, reg_strength, name='cl_i256_1')
    x = GlobalAveragePooling2D(name='cl_gap')(x)
    x = Dense(1000, kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength),name='cl_dense')(x)
    x = BatchNormalization(name='cl_last_bn')(x)
    y = ReLU()(x)
    outputs = Dense(n_classes,
                    kernel_initializer=he_normal(),name='cl')(y)
    return outputs

def resnet_mine_red(inputs, n_classes=10, reg_strength=0.001):
    x = inputs
    x = BatchNormalization(name='cl_first_bn_layer')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding="same", name='cl_max_pooling')(x)
    x = ReLU()(x)
    x = Conv2D(64,kernel_size=3,strides=1,padding='same',name='inbetween',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_s_bn_layer')(x)
    x = ReLU()(x)
    x = res_identity(x, 64, 3, reg_strength, name='cl_identity_64_1')
    x = res_identity(x, 64, 3, reg_strength, name='cl_identity_64_2')
    x = res_down(x, 128, 3, reg_strength, name='cl_first_down_128')
    x = res_identity(x, 128, 3, reg_strength, name='cl_identity_128_2')
    x = res_down(x, 256, 3, reg_strength, name='cl_second_down_256')
    x = res_identity(x, 256, 3, reg_strength, name='cl_identity_256_2')
    # x = res_identity(x, 512, 3, reg_strength, name='cl_identity_512_1')
    # x = res_identity(x, 512, 3, reg_strength, name='cl_identity_512_2')
    x = GlobalAveragePooling2D(name='cl_gap')(x)
    x = Dense(1000, kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='cl_dense')(
        x)
    x = BatchNormalization(name='cl_last_bn')(x)
    y = ReLU()(x)
    outputs = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(), name='cl')(y)
    return outputs

def licenseplate_cnn(inputs,reg_strength,n_classes):
    x = inputs
    x = Conv2D(64,kernel_size=3,strides=1,padding='same',name='cl_1_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_1_bn')(x)
    x = ReLU()(x)

    x = Conv2D(64,kernel_size=3,strides=1,padding='same',name='cl_2_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_2_bn')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_1_max')(x)

    x = Conv2D(128,kernel_size=3,strides=1,padding='same',name='cl_3_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_3_bn')(x)
    x = ReLU()(x)

    x = Conv2D(128,kernel_size=3,strides=1,padding='same',name='cl_4_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_4_bn')(x)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=1, padding="same", name='cl_2_max')(x)

    x = Conv2D(256,kernel_size=3,strides=1,padding='same',name='cl_5_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_5_bn')(x)
    x = ReLU()(x)

    x = Conv2D(256,kernel_size=3,strides=1,padding='same',name='cl_6_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_6_bn')(x)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_3_max')(x)

    x = Conv2D(512,kernel_size=3,strides=1,padding='same',name='cl_7_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_7_bn')(x)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=[1,2], padding="same", name='cl_4_max')(x)

    x = Conv2D(512, kernel_size=3, strides=1, padding='same', name='cl_8_conv',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_8_bn')(x)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_5_max')(x)
    x = Flatten()(x)
    x = Dense(256, kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='cl_1_dense')(
        x)
    x = BatchNormalization(name='cl_9_bn')(x)
    x = ReLU()(x)
    x = Dense(512, kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='cl_2_dense')(
        x)
    x = BatchNormalization(name='cl_10_bn')(x)
    x = ReLU()(x)

    output1 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_1')(x)
    output2 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_2')(x)
    output3 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_3')(x)
    output4 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_4')(x)
    output5 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_5')(x)
    output6 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_6')(x)
    output7 = Dense(n_classes,activation='softmax',
                    kernel_initializer=he_normal(),name='classification_7')(x)

    output = Concatenate(axis=1,name='cl')([output1,output2,output3,output4,output5,output6,output7])
    return output

## With dropout

def block_res_dropout(inputs,num_identity, n_filter=16, kernel_size=3,reg_strength=0.001,dropout_rate=0.5):
    x = inputs
    for i in range(num_identity):
        x = res_identity_dropout(x,n_filter=n_filter,kernel_size=kernel_size,reg_strength=reg_strength,
                                 name='sr_i{0}'.format(i),dropout_rate=dropout_rate)
    return x


def res_identity_dropout(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test',dropout_rate=0.5):
    # factor to scale the filters according to the dropout rate
    factor = (1/(1-dropout_rate))**(0.5)
    x = inputs
    x = BatchNormalization(name='{0}_bn1'.format(name))(x)
    x = ReLU()(x)
    x = Dropout(rate = dropout_rate)(x,training=True)
    x = Conv2D(int(n_filter*factor),
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)

    x = BatchNormalization(name='{0}_bn2'.format(name))(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(n_filter*factor),
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv2'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)

    return Add(name='{0}_add'.format(name))([x, inputs])



def res_down_dropout(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test',dropout_rate=0.5):
    # factor to scale the filters according to the dropout rate
    factor = (1 / (1 - dropout_rate))**(0.5)
    x = inputs
    x = BatchNormalization(name='{0}_large_bn1'.format(name))(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(n_filter*factor),
                  kernel_size=kernel_size,strides=2,
                  padding='same', name='{0}_large_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)

    x = BatchNormalization(name='{0}_large_bn2'.format(name))(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(n_filter*factor),
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_large_conv2'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)


    y = inputs
    y = BatchNormalization(name='{0}_small_bn1'.format(name))(y)
    y = ReLU()(y)
    y = Dropout(rate=dropout_rate)(y, training=True)
    y = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=2,
                  padding='same',name='{0}_small_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(y)
    return Add(name='{0}_add'.format(name))([x, y])

def licenseplate_cnn_dropout(inputs,reg_strength,dropout_rate=0.5,split_sr_from_cl=False,split = 0,n_classes=37):
    # factor to scale the filters according to the dropout rate
    factor = (1 / (1 - dropout_rate))**(0.5)
    x = inputs
    if not split_sr_from_cl or split > 1:
        x = Conv2D(int(64*factor),kernel_size=3,strides=1,padding='same',name='cl_1_conv',
                   kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
        x = BatchNormalization(name='cl_1_bn')(x)
        x = ReLU()(x)
        x = Dropout(rate=dropout_rate)(x, training=True)

    if not split_sr_from_cl or split >= 1:
        x = Conv2D(int(64*factor),kernel_size=3,strides=1,padding='same',name='cl_2_conv',
                   kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
        x = BatchNormalization(name='cl_2_bn')(x)
        x = ReLU()(x)
        x = Dropout(rate=dropout_rate)(x, training=True)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_1_max')(x)

    x = Conv2D(int(128*factor),kernel_size=3,strides=1,padding='same',name='cl_3_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_3_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = Conv2D(int(128*factor),kernel_size=3,strides=1,padding='same',name='cl_4_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_4_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = MaxPooling2D(pool_size=2, strides=1, padding="same", name='cl_2_max')(x)

    x = Conv2D(int(256*factor),kernel_size=3,strides=1,padding='same',name='cl_5_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_5_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = Conv2D(int(256*factor),kernel_size=3,strides=1,padding='same',name='cl_6_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_6_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_3_max')(x)

    x = Conv2D(int(512*factor),kernel_size=3,strides=1,padding='same',name='cl_7_conv',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_7_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = MaxPooling2D(pool_size=2, strides=[1,2], padding="same", name='cl_4_max')(x)

    x = Conv2D(int(512*factor), kernel_size=3, strides=1, padding='same', name='cl_8_conv',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_8_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_5_max')(x)
    x = Flatten()(x)

    x = Dense(int(256*factor), kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='cl_1_dense')(x)
    x = BatchNormalization(name='cl_9_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    x = Dense(int(512*factor), kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='cl_2_dense')(x)
    x = BatchNormalization(name='cl_10_bn')(x)
    x = ReLU()(x)
    x = Dropout(rate=dropout_rate)(x, training=True)

    output1 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_1')(x)
    output2 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_2')(x)
    output3 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_3')(x)
    output4 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_4')(x)
    output5 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_5')(x)
    output6 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_6')(x)
    output7 = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(),name='classification_7')(x)
    output = Concatenate(axis=1,name='cl')([output1,output2,output3,output4,output5,output6,output7])
    return output

def wdsr_dropout(inputs, scale=4,n_res_blocks=16, n_channels=3,n_filter=32,reg_strength=0.001,dropout_rate=0.5):
    # factor to scale the filters according to the dropout rate
    factor = (1 / (1 - dropout_rate))**(0.5)
    x = inputs
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(n_filter*factor), 3, padding='same', name='sr_1_conv', kernel_initializer=he_normal())(x)
    ## res blocks main branch
    m = block_res_dropout(x, num_identity=n_res_blocks, n_filter=n_filter, kernel_size=3, reg_strength=reg_strength,
                          dropout_rate=dropout_rate)
    ## Deconvolution
    m = Conv2D(3 * scale ** 2, 3, padding='same', name='sr_main_last_conv',kernel_initializer=he_normal())(m)
    m = BatchNormalization(name='sr_main_last_bn')(m)
    m = ReLU()(m)
    m = SubpixelConv2D(scale)(m)
    if n_channels == 1:
        m1 = Conv2D(1, 1, padding='same', name='sr_main_1dconv',kernel_initializer=he_normal())(m)
    else:
        m1 = m

    # skip branch
    s = Conv2D(3 * scale ** 2, 5, padding='same', name='sr_skip_last_conv',
               kernel_initializer=he_normal())(x)
    s = BatchNormalization(name='sr_skip_last_bn')(s)
    s = ReLU()(s)
    s = SubpixelConv2D(scale)(s)
    if n_channels == 1:
        s1 = Conv2D(1, 1, padding='same', name='sr_skip_1dconv',kernel_initializer=he_normal())(s)
    else:
        s1 = s

    return Add(name='sr')([m1, s1])

def fsrcnn_dropout(inputs, scale=4, n_channels=3,reg_strength=0.001,dropout_rate=0.5,split_sr_from_cl=False):
    # factor to scale the filters according to the dropout rate
    factor = (1 / (1 - dropout_rate))**(0.5)
    x = inputs
    if not split_sr_from_cl:
        ## Conv. Layer 1: feature extraction layer 1
        x = Conv2D(int(56*factor), kernel_size=5,padding='same',
                      kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_1_conv')(x)
        x = BatchNormalization(name='sr_1_bn')(x)
        x = ReLU()(x)
        x = Dropout(rate=dropout_rate)(x, training=True)

    ## Conv. Layer 2: Shrinking

    x = Conv2D(int(12*factor), kernel_size=1, padding='same',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_2_conv')(x)
    x = BatchNormalization(name='sr_2_bn')(x)
    x = ReLU()(x)

    ## Conv. Layers 3–6: Mapping
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_3_conv')(x)
    x = BatchNormalization(name='sr_3_bn')(x)
    x = ReLU()(x)

    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='sr_4_conv')(x)
    x = BatchNormalization(name='sr_4_bn')(x)
    x = ReLU()(x)

    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_5_conv',)(x)
    x = BatchNormalization(name='sr_5_bn')(x)
    x = ReLU()(x)

    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='sr_6_conv')(x)
    x = BatchNormalization(name='sr_6_bn')(x)
    x = ReLU()(x)

    ##Conv.Layer  7: Expanding
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(56*factor), kernel_size=1, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='sr_7_conv')(x)
    x = BatchNormalization(name='sr_7_bn')(x)
    x = ReLU()(x)

    ##DeConv Layer 8: Deconvolution

    x = Conv2D(3 * scale ** 2, 9, padding='same', name='sr_8_conv_up{0}'.format(scale),kernel_initializer=he_normal())(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2D(1, 1, padding='same', name='sr', kernel_initializer=he_normal())(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1

def fsrcnn_red_dropout(inputs, scale=4, n_channels=3,reg_strength=0.001,dropout_rate=0.5):
    # factor to scale the filters according to the dropout rate
    factor = (1 / (1 - dropout_rate))**(0.5)

    x = inputs

    ## Conv. Layer 2: Shrinking
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=1, padding='same',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_2_conv')(x)
    x = BatchNormalization(name='sr_2_bn')(x)
    x = ReLU()(x)

    ## Conv. Layers 3–6: Mapping
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_3_conv')(x)
    x = BatchNormalization(name='sr_3_bn')(x)
    x = ReLU()(x)

    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='sr_4_conv')(x)
    x = BatchNormalization(name='sr_4_bn')(x)
    x = ReLU()(x)

    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_5_conv')(x)
    x = BatchNormalization(name='sr_5_bn')(x)
    x = ReLU()(x)

    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(12*factor), kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_6_conv')(x)
    x = BatchNormalization(name='sr_6_bn')(x)
    x = ReLU()(x)

    ##Conv.Layer  7: Expanding
    x = Dropout(rate=dropout_rate)(x, training=True)
    x = Conv2D(int(56*factor), kernel_size=1, padding='same',
            kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),name='sr_7_conv')(x)
    x = BatchNormalization(name='sr_7_bn')(x)
    x = ReLU()(x)

    ##DeConv Layer 8: Deconvolution
    x = Conv2D(3 * scale ** 2, 9, padding='same', name='sr_8_conv_up{0}'.format(scale),
                kernel_initializer=he_normal())(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2D(1, 1, padding='same', name='sr', kernel_initializer=he_normal())(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1


### with BatchEnsemble

def block_res_be(inputs, num_identity, n_filter=16, kernel_size=3, reg_strength=0.001,ensemble_size=5):
    x = inputs
    for i in range(num_identity):
        x = res_identity_be(x, n_filter=n_filter, kernel_size=kernel_size, reg_strength=reg_strength,
                                 name='sr_i{0}'.format(i),ensemble_size=ensemble_size)
    return x


def res_identity_be(inputs, n_filter=16, kernel_size=3, reg_strength=0.001, name='test',ensemble_size=5):
    x = inputs
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='{0}_bn1'.format(name),use_tpu=False)
    x = ReLU()(x)
    x = Conv2DBatchEnsemble(
        n_filter, kernel_size=kernel_size, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
        name='{0}_conv1'.format(name), ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size, name='{0}_bn2'.format(name),use_tpu=False)
    x = ReLU()(x)
    x = Conv2DBatchEnsemble(
        n_filter, kernel_size=kernel_size, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
        name='{0}_conv2'.format(name), ensemble_size=ensemble_size)(x)
    return Add(name='{0}_add'.format(name))([x, inputs])

def res_down_be(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test',ensemble_size=5):
    random_sign_init = -0.5
    x = inputs
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size, name='{0}_large_bn1'.format(name),use_tpu=False)
    x = ReLU()(x)
    x = Conv2DBatchEnsemble(
        n_filter, kernel_size=kernel_size, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='{0}_large_conv1'.format(name), ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size, name='{0}_large_bn2'.format(name),use_tpu=False)
    x = ReLU()(x)
    x = Conv2DBatchEnsemble(
        n_filter, kernel_size=kernel_size, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='{0}_large_conv2'.format(name), ensemble_size=ensemble_size)(x)

    y = inputs
    y = ensemble_batchnorm(y,ensemble_size=ensemble_size, name='{0}_small_bn1'.format(name),use_tpu=False)
    y = ReLU()(y)
    y = Conv2DBatchEnsemble(
        n_filter, kernel_size=kernel_size, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='{0}_small_conv1'.format(name), ensemble_size=ensemble_size)(y)

    return Add(name='{0}_add'.format(name))([x, y])

def fsrcnn_red_be(inputs, scale=4, n_channels=3,reg_strength=0.001,ensemble_size=5):
    x = inputs
    random_sign_init = -0.5
    ## Conv. Layer 2: Shrinking
    x = Conv2DBatchEnsemble(
        12, kernel_size=1, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_2_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_2_bn',use_tpu=False)
    x = ReLU()(x)

    ## Conv. Layers 3–6: Mapping
    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_3_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_3_bn',use_tpu=False)
    x = ReLU()(x)


    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_4_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_4_bn',use_tpu=False)
    x = ReLU()(x)

    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_5_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_5_bn',use_tpu=False)
    x = ReLU()(x)

    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_6_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_6_bn',use_tpu=False)
    x = ReLU()(x)

    ##Conv.Layer  7: Expanding
    x = Conv2DBatchEnsemble(
        56, kernel_size=1, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_^7_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_7_bn',use_tpu=False)
    x = ReLU()(x)

    ##DeConv Layer 8: Deconvolution
    x = Conv2DBatchEnsemble(
        3 * scale ** 2, kernel_size=9, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_8_conv_up{0}'.format(scale), ensemble_size=ensemble_size)(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2DBatchEnsemble(
            1, kernel_size=1, use_bias=False, padding='same', strides=1,
            kernel_regularizer=regularizers.l2(reg_strength),
            alpha_initializer=make_random_sign_initializer(random_sign_init),
            gamma_initializer=make_random_sign_initializer(random_sign_init),
            name='sr', ensemble_size=ensemble_size)(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1

def fsrcnn_be(inputs, scale=4, n_channels=3,reg_strength=0.001,ensemble_size=5,split_sr_from_cl=False):
    x = inputs
    random_sign_init = -0.5
    if not split_sr_from_cl:
    ## Conv. Layer 1: feature extraction layer 1
        x = Conv2DBatchEnsemble(
            56, kernel_size=5, use_bias=False, padding='same', strides=1,
            kernel_regularizer= regularizers.l2(reg_strength),
            alpha_initializer=make_random_sign_initializer(random_sign_init),
            gamma_initializer=make_random_sign_initializer(random_sign_init),
            name='sr_2_conv', ensemble_size=ensemble_size)(x)
        x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_1_bn',use_tpu=False)
        x = ReLU()(x)

    ## Conv. Layer 2: Shrinking
    x = Conv2DBatchEnsemble(
        12, kernel_size=1, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_2_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_2_bn',use_tpu=False)
    x = ReLU()(x)

    ## Conv. Layers 3–6: Mapping
    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_3_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_3_bn',use_tpu=False)
    x = ReLU()(x)


    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_4_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_4_bn',use_tpu=False)
    x = ReLU()(x)

    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_5_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_5_bn',use_tpu=False)
    x = ReLU()(x)

    x = Conv2DBatchEnsemble(
        12, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_6_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_6_bn',use_tpu=False)
    x = ReLU()(x)

    ##Conv.Layer  7: Expanding
    x = Conv2DBatchEnsemble(
        56, kernel_size=1, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_7_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='sr_7_bn',use_tpu=False)
    x = ReLU()(x)

    ##DeConv Layer 8: Deconvolution
    x = Conv2DBatchEnsemble(
        3 * scale ** 2, kernel_size=9, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='sr_8_conv_up{0}'.format(scale), ensemble_size=ensemble_size)(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2DBatchEnsemble(
            1, kernel_size=1, use_bias=False, padding='same', strides=1,
            kernel_regularizer=regularizers.l2(reg_strength),
            alpha_initializer=make_random_sign_initializer(random_sign_init),
            gamma_initializer=make_random_sign_initializer(random_sign_init),
            name='sr', ensemble_size=ensemble_size)(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1


def wdsr_be(inputs, scale=4,n_res_blocks=16, n_channels=3,n_filter=32,reg_strength=0.001,ensemble_size=5):
    x = inputs
    x = Conv2DBatchEnsemble(
        n_filter, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
        name='sr_1_conv', ensemble_size=ensemble_size)(x)

    ## res blocks main branch
    m = block_res_be(x, num_identity=n_res_blocks, n_filter=n_filter, kernel_size=3,
                     reg_strength=reg_strength,ensemble_size=ensemble_size)

    ## Deconvolution
    m = Conv2DBatchEnsemble(
        3 * scale ** 2, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
        name='sr_main_last_conv', ensemble_size=ensemble_size)(m)
    m = ensemble_batchnorm(m,ensemble_size=ensemble_size,name='sr_main_last_bn',use_tpu=False)
    m = ReLU()(m)
    m = SubpixelConv2D(scale)(m)
    if n_channels == 1:
        m1 = Conv2DBatchEnsemble(
            1, kernel_size=1, use_bias=False, padding='same', strides=1,
            kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer='he_normal',
            alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
            name='sr_main_1dconv', ensemble_size=ensemble_size)(m)
    else:
        m1 = m

    # skip branch
    s = Conv2DBatchEnsemble(
        3 * scale ** 2, kernel_size=5, use_bias=False, padding='same', strides=1,
        kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer='he_normal',
        alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
        name='sr_skip_last_conv', ensemble_size=ensemble_size)(x)
    s = ensemble_batchnorm(s,ensemble_size=ensemble_size,name='sr_skip_last_bn',use_tpu=False)
    s = ReLU()(s)
    s = SubpixelConv2D(scale)(s)
    if n_channels == 1:
        s1 = Conv2DBatchEnsemble(
            1, kernel_size=1, use_bias=False, padding='same', strides=1,
            kernel_regularizer=regularizers.l2(reg_strength), kernel_initializer='he_normal',
            alpha_initializer=make_random_sign_initializer(3), gamma_initializer=make_random_sign_initializer(3),
            name='sr_skip_1dconv', ensemble_size=ensemble_size)(s)
    else:
        s1 = s

    return Add(name='sr')([m1, s1])

def licenseplate_cnn_be(inputs,reg_strength,ensemble_size=5,split_sr_from_cl=False,split = 0,n_classes=37):
    x = inputs
    random_sign_init = -0.5
    if not split_sr_from_cl or split > 1:
        x = Conv2DBatchEnsemble(
            64, kernel_size=3, use_bias=False, padding='same', strides=1,
            kernel_regularizer= regularizers.l2(reg_strength),
            alpha_initializer=make_random_sign_initializer(random_sign_init),
            gamma_initializer=make_random_sign_initializer(random_sign_init),
            name='cl_1_conv', ensemble_size=ensemble_size)(x)
        x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_1_bn',use_tpu=False)
        x = ReLU()(x)
    if not split_sr_from_cl or split >= 1:
        x = Conv2DBatchEnsemble(
            64, kernel_size=3, use_bias=False, padding='same', strides=1,
            kernel_regularizer= regularizers.l2(reg_strength),
            alpha_initializer=make_random_sign_initializer(random_sign_init),
            gamma_initializer=make_random_sign_initializer(random_sign_init),
            name='cl_2_conv', ensemble_size=ensemble_size)(x)
        x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_2_bn',use_tpu=False)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_1_max')(x)

    x = Conv2DBatchEnsemble(
        128, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='cl_3_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_3_bn',use_tpu=False)
    x = ReLU()(x)

    x = Conv2DBatchEnsemble(
        128, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='cl_4_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_4_bn',use_tpu=False)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=1, padding="same", name='cl_2_max')(x)

    x = Conv2DBatchEnsemble(
        256, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='cl_5_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_5_bn',use_tpu=False)
    x = ReLU()(x)

    x = Conv2DBatchEnsemble(
        256, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='cl_6_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_6_bn',use_tpu=False)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_3_max')(x)

    x = Conv2DBatchEnsemble(
        512, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='cl_7_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_7_bn',use_tpu=False)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=[1,2], padding="same", name='cl_4_max')(x)

    x = Conv2DBatchEnsemble(
        512, kernel_size=3, use_bias=False, padding='same', strides=1,
        kernel_regularizer= regularizers.l2(reg_strength),
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        name='cl_8_conv', ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_8_bn',use_tpu=False)
    x = ReLU()(x)

    x = MaxPooling2D(pool_size=2, strides=2, padding="same", name='cl_5_max')(x)
    x = Flatten()(x)
    x = DenseBatchEnsemble(
        1024,
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        activation=None, name='cl_1_dense',
        kernel_regularizer=regularizers.l2(reg_strength),
        bias_regularizer=regularizers.l2(reg_strength),
        ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_9_bn',use_tpu=False)
    x = ReLU()(x)
    x = DenseBatchEnsemble(
        2048,
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
        activation=None, name='cl_2_dense',
        kernel_regularizer=regularizers.l2(reg_strength),
        bias_regularizer=regularizers.l2(reg_strength),
        ensemble_size=ensemble_size)(x)
    x = ensemble_batchnorm(x,ensemble_size=ensemble_size,name='cl_10_bn',use_tpu=False)
    x = ReLU()(x)

    output1 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_1', ensemble_size=ensemble_size)(x)
    output2 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_2', ensemble_size=ensemble_size)(x)
    output3 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_3', ensemble_size=ensemble_size)(x)
    output4 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_4', ensemble_size=ensemble_size)(x)
    output5 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_5', ensemble_size=ensemble_size)(x)
    output6 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_6', ensemble_size=ensemble_size)(x)
    output7 = DenseBatchEnsemble(n_classes,activation='softmax',
        alpha_initializer=make_random_sign_initializer(random_sign_init),
        gamma_initializer=make_random_sign_initializer(random_sign_init),
                                 name='classification_7',  ensemble_size=ensemble_size)(x)


    output = Concatenate(axis=1,name='cl')([output1,output2,output3,output4,output5,output6,output7])

    return output
