from utils.callback import ModelCheckpointAfter,GradNorm
from tensorflow.keras.callbacks import  TensorBoard,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.optimizers import Adam
from contextlib import redirect_stdout


import os
class Trainer:
    def __init__(self,training,model,store):
        self.training = training
        self.model = model
        self.callbacks = []
        self.callbacks_all = {}
        self.store = store
        self.create_train_workspace()
        self.def_callbacks()
        self.write_summary()



    def create_train_workspace(self):

        os.makedirs(self.store["dir"], exist_ok=True)

        models_dir = os.path.join(self.store["dir"], 'models')
        log_dir = os.path.join(self.store["dir"], 'log')

        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.models_dir = models_dir
        self.log_dir = log_dir

        return

    def write_summary(self):
        with open(os.path.join(self.store["dir"], 'summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

    def compile_model(self,loss,loss_weights,metric):
        if self.model.name == "cl":
            self.model.compile(optimizer=Adam(lr=self.training["lr"]),
                               loss = loss['cl'],
                               metrics = [metric['cl']])
            return

        if self.model.name == "sr":
            self.model.compile(optimizer=Adam(lr=self.training["lr"]),
                               loss = loss['sr'],
                               metrics = [metric['sr']])
            return

        self.model.compile(optimizer=Adam(lr=self.training["lr"]),
                           loss = loss,
                           loss_weights = loss_weights,
                           metrics = metric)
        return

    def set_callbacks(self,list):
        for i in range(len(list)):
            if list[i] == "gradNorm":
                self.custom_callbacks = self.callbacks_all[list[i]]
                self.custom = True
            else:
                self.callbacks.append(self.callbacks_all[list[i]])

        del self.callbacks_all

    def def_callbacks(self):
        self.callbacks_all["tb"] = TensorBoard(log_dir=self.log_dir,
                                               write_graph=False, write_grads=False )
        self.callbacks_all["reduceLR"] = ReduceLROnPlateau(monitor=self.training["monitor"],
                                                           patience=self.training["patience"],
                                                           factor=self.training["lr_decay"],
                                                           min_lr=self.training["lr_min"])
        self.callbacks_all["earlyStop"] = EarlyStopping(monitor=self.training["monitor"],
                                                        patience= self.training["patience"],
                                                        mode='min')
        p = self.store["period"]
        if self.store["best"]:
            p = 1


        self.callbacks_all["store"] = ModelCheckpointAfter(0, filepath=os.path.join(self.models_dir,'epoch-best.h5'),
                                                           monitor=self.training["monitor"],
                                                           save_best_only=self.store["best"],
                                                           mode='min', period=p,
                                                           save_weights_only=True,
                                                           verbose=1)
        self.callbacks_all["gradNorm"] = GradNorm(rate=self.training["gn_rate"],
                                                  alpha=self.training["gn_alpha"],
                                                  T=self.training["gn_T"])
        return

    def train_generator(self,generator, X_lr_val, X_hr_val,l_val,n_batch):


        if self.model.name == "cl":

            self.model.fit(x=generator,
                           batch_size = n_batch,
                           epochs=self.training["epochs"],
                           initial_epoch=self.training["epoch_init"],
                           validation_data=(X_lr_val, l_val),
                           verbose=1,
                           shuffle=True,
                           callbacks=self.callbacks,
                           workers=7
                           )
            return

        if self.model.name == "sr":
            self.model.fit(x=generator,
                           batch_size = n_batch,
                           epochs=self.training["epochs"],
                           initial_epoch=self.training["epoch_init"],
                           validation_data=(X_lr_val, X_hr_val),
                           verbose=1,
                           shuffle=True,
                           callbacks=self.callbacks)
            return


        self.model.fit(x=generator,
                       batch_size = n_batch,
                       epochs=self.training["epochs"],
                       initial_epoch=self.training["epoch_init"],
                       validation_data=(X_lr_val, [X_hr_val, l_val]),
                       verbose=1,
                       shuffle=True,
                       callbacks=self.callbacks)
        return

    def train(self,X_lr_train, X_hr_train,l_train,X_lr_val, X_hr_val,l_val,n_batch):
        if self.model.name == "cl":
            self.model.fit(x = X_lr_train, y = l_train,
                        epochs=self.training["epochs"],
                        initial_epoch=self.training["epoch_init"] ,
                        batch_size = n_batch,
                        validation_data= (X_lr_val,l_val),
                        verbose = 2,
                        shuffle=True,
                        callbacks=self.callbacks)
            return

        if self.model.name == "sr":
            self.model.fit(x = X_lr_train, y = X_hr_train,
                        epochs=self.training["epochs"],
                        initial_epoch=self.training["epoch_init"] ,
                        batch_size= n_batch,
                        validation_data= (X_lr_val,X_hr_val),
                        verbose = 2,
                        shuffle=True,
                        callbacks=self.callbacks)
            return


        self.model.fit(x = X_lr_train, y = [X_hr_train,l_train],
                    epochs=self.training["epochs"],
                    initial_epoch=self.training["epoch_init"] ,
                    batch_size= n_batch,
                    validation_data= (X_lr_val,[X_hr_val,l_val]),
                    verbose = 2,
                    shuffle=True,
                    callbacks=self.callbacks)
        return

