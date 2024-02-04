import os
import time
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.backend import gradients
from keras import backend as K
from keras.callbacks import Callback
import numpy as np


class ModelCheckpointAfter(ModelCheckpoint):
    def __init__(self, epoch, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):

        super().__init__(filepath=filepath, monitor=monitor,verbose= verbose, save_best_only=save_best_only,
                         save_weights_only=save_weights_only, mode=mode, save_freq='epoch')
        self.after_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.after_epoch:
            super().on_epoch_end(epoch, logs)


def learning_rate(step_size, decay, verbose=1):
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr

    return LearningRateScheduler(schedule, verbose=verbose)



class MGDA(Callback):
    def __init__(self,alpha):
        
        self.num = 0
        self.m_update_sr = None
        self.m_update_cl = None
        self.alpha = K.variable(value=0)


    def l2norm(self,input):
        return K.sqrt(K.sum((input)**2))**2

    def custom_on_train_begin(self):
        ## SR loss

        last_common = self.model.get_layer("last_common")
        conv_last_common = last_common._trainable_weights[0]

        tmp_G1 = K.flatten(gradients(self.model.metrics_tensors[0], conv_last_common))
        flatten_G1 = K.reshape(tmp_G1,[-1,1])

        ## Cl loss
        tmp_G2 = K.flatten(gradients(self.model.metrics_tensors[1], conv_last_common))
        flatten_G2= K.reshape(tmp_G2, [-1, 1])
        Denom = self.l2norm(flatten_G1 - flatten_G2)
        Nom = K.dot(K.transpose(flatten_G2 - flatten_G1),flatten_G2)

        self.alpha = Nom/Denom


    def custom_on_batch_begin(self,x,y):
        sess = K.get_session()

        a = sess.run(self.alpha, feed_dict={self.model.inputs[0]: x,
                                      self.model.targets[0]: y[0],
                                      self.model.targets[1]: y[1],
                                      self.model.sample_weights[0]: np.ones(y[1].shape[0]),
                                      self.model.sample_weights[1]: np.ones(y[1].shape[0])})

        a = max(min(a,1),0)

        if a == 0:
            K.set_value(self.model.loss_weights[0], 0 )
            K.set_value(self.model.loss_weights[1], 1 )
        elif a == 1:
            K.set_value(self.model.loss_weights[0], 1 )
            K.set_value(self.model.loss_weights[1], 0)
        else:
            K.set_value(self.model.loss_weights[0], a[0,0])
            K.set_value(self.model.loss_weights[1], 1 - a[0,0])

    def custom_on_batch_end(self, x, y, outs, batch, logs):
        pass

    def custom_on_epoch_end(self, x, y, outs, epoch, logs):
        pass

class StoreWeights(Callback):
    def __init__(self):
        self.i = 0
        self.lenW = 0
        self.f = open('weightUpdates.csv','w')

    def custom_on_train_begin(self):
        for i in range(len(self.model.trainable_weights)):
            self.f.write("{0};".format(self.model.trainable_weights[i].name))
        self.f.write("\n")


    def custom_on_batch_begin(self,x,y):
        pass

    def custom_on_batch_end(self, x, y, outs, batch, logs):
        if self.i == 0:
            self.lenW = len(self.model.trainable_weights)
            self.w_current = [None] * self.lenW

            for i in range(self.lenW):
                self.w_current[i] = K.get_value(self.model.trainable_weights[i])
            self.i = 1
        else:

            w_new = [None] * self.lenW

            for i in range(self.lenW):
                w_new[i] = K.get_value(self.model.trainable_weights[i])
                update = self.w_current[i] - w_new[i]
                nom = np.linalg.norm(update.ravel())
                denom = np.linalg.norm(self.w_current[i].ravel())

                self.f.write("{0:.05};".format(nom / denom))

            self.f.write("\n")
            self.w_current = w_new
            # param_scale = np.linalg.norm(self.model.optimizer.updates[1])


    def custom_on_epoch_end(self, x, y, outs, epoch, logs):
        self.f.close()
        pass

class Empty(Callback):
    def __init__(self):
        pass
    def custom_on_train_begin(self):
        pass

    def custom_on_batch_begin(self,x,y):
        pass

    def custom_on_batch_end(self, x, y, outs, batch, logs):
        pass

    def custom_on_epoch_end(self, x, y, outs, epoch, logs):
        pass


class GradNorm(Callback):
    def __init__(self,rate,alpha,T):
        self.T = T
        self.alpha = alpha
        self.L0_sr = K.variable(value=0)
        self.L0_cl = K.variable(value=0)
        self.state = 0
        self.lr = rate
        self.m_update_sr = None
        self.m_update_cl = None
        self.currentWeightSR = K.variable(value=0)
        self.currentWeightCl = K.variable(value=0)

    def l2norm(self,input):
        return K.sqrt(K.sum((input)**2))

    def custom_on_batch_begin(self, x, y):
        pass

    def custom_on_train_begin(self):
        K.set_value(self.currentWeightSR,K.get_value(self.model.loss_weights[0]))
        last_common = self.model.get_layer("last_common")
        conv_last_common = last_common._trainable_weights[0]

        flatten_G1 = K.flatten(gradients(self.currentWeightSR * self.model.metrics_tensors[0],
                                         conv_last_common))
        G_sr = self.l2norm(flatten_G1)

        ##Cl loss
        K.set_value(self.currentWeightCl, K.get_value(self.model.loss_weights[1]))
        flatten_G2 = K.flatten(gradients(self.currentWeightCl * self.model.metrics_tensors[1],
                                         conv_last_common))
        G_cl = self.l2norm(flatten_G2)

        G_mean = (G_sr + G_cl) * 0.5

        L_sr = self.model.metrics_tensors[0] / self.L0_sr
        L_cl = self.model.metrics_tensors[1] / self.L0_cl
        L_mean = (L_sr + L_cl) * 0.5

        C_sr = G_mean * ((L_sr / L_mean) ** self.alpha)
        C_cl = G_mean * ((L_cl / L_mean) ** self.alpha)

        part1 = K.sqrt(K.sum(K.abs((self.l2norm(self.model.loss_weights[0] * flatten_G1)) - C_sr)))
        part2 = K.sqrt(K.sum(K.abs((self.l2norm(self.model.loss_weights[1] * flatten_G2)) - C_cl)))

        Lgrad = K.sum(part1 + part2)

        self.m_update_sr = gradients(Lgrad, self.model.loss_weights[0])
        self.m_update_cl = gradients(Lgrad, self.model.loss_weights[1])


    def custom_on_epoch_end(self,x,y,outs,epoch,logs):

        last_common = self.model.get_layer("last_common")
        conv_last_common = last_common._trainable_weights[0]
        # if epoch == 0:
        #     self.state = 1
        #     K.set_value(self.L0_sr,K.get_value(self.L0_sr))
        #     K.set_value(self.L0_cl, K.get_value(self.L0_cl))

        if self.state == 0:
            self.L0_sr = logs['sr_loss']
            self.L0_cl = logs['cl_loss']
            self.state = 1
        else:

            ## SR loss
            flatten_G1 = K.flatten(gradients(K.get_value(self.model.loss_weights[0]) * self.model.metrics_tensors[0],conv_last_common))
            G_sr = self.l2norm(flatten_G1)

            ##Cl loss
            flatten_G2 = K.flatten(gradients(K.get_value(self.model.loss_weights[1])*self.model.metrics_tensors[1], conv_last_common))
            G_cl =self.l2norm(flatten_G2)

            G_mean = (G_sr+G_cl)*0.5

            L_sr = logs['sr_loss']/self.L0_sr
            L_cl = logs['cl_loss'] / self.L0_cl
            L_mean = (L_sr + L_cl)*0.5
            R_sr = (L_sr / L_mean)**self.alpha
            R_cl = (L_cl / L_mean)**self.alpha

            C_sr = G_mean*R_sr
            C_cl = G_mean*R_cl

            part1 = K.sqrt(K.sum(K.abs((self.l2norm(self.model.loss_weights[0]*flatten_G1)) - C_sr)))
            part2 = K.sqrt(K.sum(K.abs((self.l2norm(self.model.loss_weights[1]*flatten_G2)) - C_cl)))

            Lgrad = K.sum(part1 + part2)

            m_update_sr = gradients(Lgrad,self.model.loss_weights[0])
            m_update_cl = gradients(Lgrad, self.model.loss_weights[1])

            sess = K.get_session()
            up_sr = sess.run(m_update_sr[0], feed_dict={self.model.inputs[0]: x,
                                          self.model.targets[0]: y[0],
                                          self.model.targets[1]: y[1],
                                          self.model.sample_weights[0]: np.ones(y[1].shape[0]),
                                          self.model.sample_weights[1]: np.ones(y[1].shape[0])})

            up_cl = sess.run(m_update_cl[0], feed_dict={self.model.inputs[0]: x,
                                          self.model.targets[0]: y[0],
                                          self.model.targets[1]: y[1],
                                          self.model.sample_weights[0]: np.ones(y[1].shape[0]),
                                          self.model.sample_weights[1]: np.ones(y[1].shape[0])})

            new_w_sr = K.get_value(self.model.loss_weights[0]) - self.lr*up_sr
            new_w_cl = K.get_value(self.model.loss_weights[1]) - self.lr*up_cl

            new_w_sr = max(new_w_sr,0.0)
            new_w_cl = max(new_w_cl, 0.0)

            sum_new = new_w_sr + new_w_cl

            new_w_sr = (self.T / sum_new) * new_w_sr
            new_w_cl = (self.T / sum_new) * new_w_cl
            K.set_value(self.model.loss_weights[0], new_w_sr)
            K.set_value(self.model.loss_weights[1], new_w_cl)

        print('Epoch {0}  new SR weight {1:.5f}    new Cl weight {2:.5f}'.format(epoch,K.get_value(self.model.loss_weights[0]),K.get_value(self.model.loss_weights[1])))


    def custom_on_batch_end(self,x,y,outs,batch,logs):

        if self.state == 0:
            #self.num += 1
            K.set_value(self.L0_sr,K.get_value(self.L0_sr) + logs['sr_loss'])
            K.set_value(self.L0_cl, K.get_value(self.L0_cl) + logs['cl_loss'])
            self.state = 1

        else:
            #tmp = tape.gradient(K.get_value(self.model.loss_weights[0]) * self.model.metrics_tensors[0],self.model.trainable_weights[self.split*6])
            ## SR loss

            K.set_value(self.currentWeightCl, K.get_value(self.model.loss_weights[1]))
            K.set_value(self.currentWeightSR, K.get_value(self.model.loss_weights[0]))
            sess = K.get_session()


            up_sr = sess.run(self.m_update_sr[0], feed_dict={self.model.inputs[0]: x,
                                          self.model.targets[0]: y[0],
                                          self.model.targets[1]: y[1],
                                          self.model.sample_weights[0]: np.ones(y[1].shape[0]),
                                          self.model.sample_weights[1]: np.ones(y[1].shape[0])})

            up_cl = sess.run(self.m_update_cl[0], feed_dict={self.model.inputs[0]: x,
                                          self.model.targets[0]: y[0],
                                          self.model.targets[1]: y[1],
                                          self.model.sample_weights[0]: np.ones(y[1].shape[0]),
                                          self.model.sample_weights[1]: np.ones(y[1].shape[0])})



            new_w_sr = K.get_value(self.model.loss_weights[0]) - self.lr*up_sr
            new_w_cl = K.get_value(self.model.loss_weights[1]) - self.lr*up_cl

            new_w_sr = max(new_w_sr, 0.0)
            new_w_cl = max(new_w_cl, 0.0)

            sum_new = new_w_sr + new_w_cl

            new_w_sr = (self.T / sum_new) * new_w_sr
            new_w_cl = (self.T / sum_new) * new_w_cl

            #print('Batch {0} up SR weight {1:.5f}  new SR weight {3:.5f}   up Cl weight {2:.5f}  new Cl weight {4:.5f}'.format(batch, up_sr, up_cl,new_w_sr,new_w_cl))

            K.set_value(self.model.loss_weights[0], new_w_sr)
            K.set_value(self.model.loss_weights[1], new_w_cl)

