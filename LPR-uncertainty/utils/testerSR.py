from sklearn.metrics import precision_recall_curve,roc_curve,auc, confusion_matrix, accuracy_score, f1_score, multilabel_confusion_matrix
import numpy as np
import json
import tensorflow as tf
#from skimage.measure import compare_ssim as ssim

from skimage.metrics import structural_similarity as ssim
from statistics import mean, stdev
import pandas as pd
from architecture.model import BuildModel
import os
import matplotlib.pyplot as plt
import csv
import datetime
import random

class TesterSR:
    def __init__(self,folder,epoch,data):
        self.folder = folder
        self.epoch = epoch

        self.data = data
        if self.data["dataset"] == "czech":
            self.n_classes = 37
        elif self.data["dataset"] == "ccpd":
            self.n_classes = 68
        self.dictLP = ["0","1","2","3","4","5","6","7","8","9"
                       ,"A","B","C","D","E","F","G","H","I","J","K","L","M",
                       "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
                       ,""]

    def create_dataframe(self):
        exits = True
        if self.data["noise"]:
            self.logs_file = os.path.join(self.folder, "noise_sr.csv")
            if not os.path.exists(self.logs_file):
                exits = False
                self.df = pd.DataFrame()
                self.df["noiseType"] = []
                self.df["std"] = []
                self.df["n_inference"] = []

        elif self.data["blur"]:
            self.logs_file = os.path.join(self.folder, "blur_sr.csv")
            if not os.path.exists(self.logs_file):
                exits = False
                self.df = pd.DataFrame()
                self.df["blurType"] = []
                self.df["strength"] = []
                self.df["kernel"] = []
                self.df["n_inference"] = []
        else:
            self.logs_file = os.path.join(self.folder, "normal_sr.csv")
            if not os.path.exists(self.logs_file):
                self.df = pd.DataFrame()
                self.df["n_inference"] = []
                exits = False

        if exits:
            self.df = pd.read_csv(self.logs_file,sep=';')
            return


        self.df["PSNR"] = []
        self.df["MS-SSIM"] = []



    def load_pretrained(self):
        # load the information regarding the network from the json file
        total = json.load(open( os.path.join(self.folder,"dict.json"), 'r' ))
        self.store = total["store"]
        self.cl_net = total["design"]["cl_net"]
        total["design"]["model_type"] = "sr"
        self.n_ensemble = total["parameter"]["ensemble_size"]
        # get the name of the pretrained model
        model_path = os.path.join(self.folder,"models")
        if self.epoch > 0:
            model_idx = 'epoch-{0:03d}.h5'.format(self.epoch)
        else:
            # if no epoch is give, get the last one
            model_idx = os.listdir(model_path)[-1]
        # build the model
        builder = BuildModel(data=total["data"], parameter=total["parameter"], design=total["design"])
        self.model = builder.setup_model()
        # load the pretrained weights

        self.model.load_weights(os.path.join(model_path, model_idx),by_name=True)

    def lp_to_str(self,lp_pred):
        str = ""

        for i in range(7):
            str += self.dictLP[lp_pred[ i *  self.n_classes:(i + 1) *  self.n_classes].argmax()]
        return str


    def batch_ssim(self,X_hr_test, X_hr_pred):
        assert (X_hr_pred.shape == X_hr_test.shape)
        ssim_all =  tf.image.ssim(X_hr_pred, X_hr_test,max_val= 1)

        ssim_all_m = mean(ssim_all.numpy())
        #ssim_all_s = stdev(ssim_all.numpy())
        #print(f"SSIM {ssim_all_m}")
        #self.file.write(f"SSIM {ssim_all_m} \n")

        self.ssim_all_m = ssim_all_m
        #self.ssim_all_s = ssim_all_s

    def batch_psnr(self,X_hr_test, X_hr_pred):
        assert (X_hr_pred.shape == X_hr_test.shape)
        psnr_all = tf.image.psnr(X_hr_pred, X_hr_test,max_val= 1)

        psnr_all_m = mean(psnr_all.numpy())
        #psnr_all_s = stdev(psnr_all.numpy())
        #print(f"SSIM {ssim_all_m}")
        #self.file.write(f"SSIM {ssim_all_m} \n")

        self.psnr_all_m = psnr_all_m
        #self.psnr_all_s = psnr_all_s



    def test_dropout_example(self,X_lr_test, X_hr_test,l_test,name,num):

        if self.model.name == "cl":
            return

        elif self.model.name == "sr":
            return
        else:

            X_hr_pred = np.zeros((num,X_hr_test.shape[0],X_hr_test.shape[1],X_hr_test.shape[2],X_hr_test.shape[3]),dtype=np.float32)
            l_pred = np.zeros((num, l_test.shape[0], l_test.shape[1]), dtype=np.float32)
            for i in range(num):
                X_hr_pred[i,], l_pred[i,] = self.model.predict(x=X_lr_test)

            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            # compute mean image
            mean_img = np.mean(X_hr_pred, axis=0)


            # compute mean lp
            mean_lp = np.mean(l_pred, axis=0)

            lp_str = [self.lp_to_str(mean_lp[i,]) for i in range(mean_lp.shape[0])]
            lp_str_gt = [self.lp_to_str(l_test[i,]) for i in range(l_test.shape[0])]
        return mean_img,lp_str,lp_str_gt

    def test_dropout(self,X_lr_test, X_hr_test,num):

        if self.model.name == "cl":
            return

        elif self.model.name == "sr":

            mean_img = np.zeros(X_hr_test.shape).astype(np.float32)
            for i in range(num):
                mean_img += (1/num)* self.model.predict(x = X_lr_test).astype(np.float32)

            mean_img[mean_img < 0] = 0
            mean_img[mean_img > 1] = 1



            # Comptue MS-SSIM and PSNR of the batch
            self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            self.batch_psnr(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            # Show the results
            # self.show_img(X_lr_test, X_hr_test, l_test, mean_img, std_img,name)
            # Store the results
            if self.data["noise"]:
                if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                               (self.df['std'] == self.data["noise_low"]) &
                               (self.df['n_inference']== num)])  == 0:

                    self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                              'n_inference': num, 'MS-SSIM':self.ssim_all_m,
                                              'PSNR':self.psnr_all_m}, ignore_index=True)

            elif self.data["blur"]:
                if len(self.df[(self.df["blurType"] == self.data["blurType"]) &
                               (self.df['strength'] == self.data["blur_strength"]) &
                               (self.df['kernel'] == self.data["blur_kernel"]) &
                               (self.df['n_inference'] == num)]) == 0:
                    self.df = self.df.append({'blurType': self.data["blurType"], 'strength': self.data["blur_strength"],
                                              'kernel': self.data["blur_kernel"],'n_inference': num,
                                              'MS-SSIM':self.ssim_all_m, 'PSNR':self.psnr_all_m}, ignore_index=True)
        else:

            X_hr_pred = np.zeros((num,X_hr_test.shape[0],X_hr_test.shape[1],X_hr_test.shape[2],X_hr_test.shape[3]),dtype=np.float32)
            for i in range(num):
                X_hr_pred[i,],X_pred = self.model.predict(x = X_lr_test)


            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            # compute mean image
            mean_img = np.mean(X_hr_pred,axis=0)
            # compute uncertainty
            std_img = np.std(X_hr_pred,axis=0)
            # compute mean lp

            # for one image, compute the top3 license plates and store in csv file to provide an example
            #name_top3,lp_top3,max_top3,std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,],lp_std=std_lp[0,],k=3)
            #self.write_dropout(name,lp_top3,max_top3,std_top3,num)
            # Comptue SSIM of the batch
            self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            self.batch_psnr(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            # show results
            # self.show_img_lp(X_lr_test, X_hr_test, l_test, mean_img, std_img, name, mean_lp, std_lp, num)
            # store results
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                            (self.df['std'] == self.data["noise_low"]) &
                            (self.df['n_inference']== num)])  == 0:

                self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                           'MS-SSIM':self.ssim_all_m, 'PSNR':self.psnr_all_m}, ignore_index=True)
        self.df.to_csv(self.logs_file,sep=';', index=False)



    def test(self,X_lr_test, X_hr_test):

        if self.model.name == "cl":

            return


        elif self.model.name == "sr":

            X_hr_pred = self.model.predict(x = X_lr_test)
            X_hr_pred = X_hr_pred.astype(np.float32)
            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            self.batch_ssim(X_hr_test=X_hr_test,X_hr_pred=X_hr_pred)
            self.batch_psnr(X_hr_test=X_hr_test, X_hr_pred=X_hr_pred)
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference']== 50)])  == 0:

                self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                           'MS-SSIM':self.ssim_all_m, 'PSNR':self.psnr_all_m}, ignore_index=True)

        else:

            X_hr_pred, l_pred = self.model.predict(x = X_lr_test)
            X_hr_pred = X_hr_pred.astype(np.float32)
            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            self.batch_ssim(X_hr_test=X_hr_test,X_hr_pred=X_hr_pred)
            self.batch_psnr(X_hr_test=X_hr_test, X_hr_pred=X_hr_pred)
 
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference'] == 50)]) == 0:
                self.df = self.df.append({'noiseType': self.data["noiseType"], 'std': self.data["noise_low"],
                                          'MS-SSIM':self.ssim_all_m, 'PSNR':self.psnr_all_m},
                                         ignore_index=True)
        #self.df.to_csv(self.logs_file, sep=';', index=False)

    def test_ensemble(self,X_lr_test):
        if self.model.name == "cl":

            l_pred = self.model.predict(x=X_lr_test)


        elif self.model.name == "sr":

            X_hr_pred = self.model.predict(x=X_lr_test)

        else:

            X_hr_pred, l_pred = self.model.predict(x=X_lr_test)

        return l_pred

    def evaluate_ensemble(self,l_pred,l_test,num):

        if self.model.name == "cl":

            # compute mean lp
            mean_lp = np.mean(l_pred,axis=0)
            # compute uncertainty measure
            std_lp = np.std(l_pred,axis=0)

            #evaluate the accuracy of the mean prediction
            self.evaluate_lp(l_test=l_test, l_pred=mean_lp)
            # evalaute the uncertainty of the prediction
            self.evaluate_lp_uncertainty( l_test=l_test,l_pred = mean_lp,std= std_lp)

            # for one image, compute the top3 license plates and store in csv file to provide an example
            #name_top3,lp_top3,max_top3,std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,],lp_std=std_lp[0,],k=3)
            #self.write_dropout(name,lp_top3,max_top3,std_top3,num)
            # Show the results
            # self.show_lp(X_lr_test, l_test, name, num)
            # Store the results

            # make sure the row does not exist

            if self.data["noise"]:
                if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                               (self.df['std'] == self.data["noise_low"]) &
                               (self.df['n_inference'] == num)]) == 0:
                    self.df = self.df.append({'noiseType': self.data["noiseType"], 'std': self.data["noise_low"],
                                              'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                              'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                              'AUC_pr': self.auc_pr,'AUC_roc': self.auc_roc,
                                               'min_uC': self.unc_c_box[0],'q1_uC': self.unc_c_box[1],
                                              'med_uC': self.unc_c_box[2],'q3_uC': self.unc_c_box[3],
                                              'max_uC': self.unc_c_box[4],'min_uF': self.unc_f_box[0],
                                              'q1_uF': self.unc_f_box[1],'med_uF': self.unc_f_box[2],
                                              'q3_uF': self.unc_f_box[3], 'max_uF': self.unc_f_box[4],'n_inference': num},
                                             ignore_index=True)
            if self.data["blur"]:
                if len(self.df[(self.df["blurType"] == self.data["blurType"]) &
                               (self.df['strength'] == self.data["blur_strength"]) &
                               (self.df['kernel'] == self.data["blur_kernel"]) &
                               (self.df['n_inference'] == num)]) == 0:
                    self.df = self.df.append({'blurType': self.data["blurType"], 'strength': self.data["blur_strength"],
                                              'kernel': self.data["blur_kernel"],
                                              'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                              'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                              'AUC_pr': self.auc_pr,'AUC_roc': self.auc_roc,
                                              'min_uC': self.unc_c_box[0],'q1_uC': self.unc_c_box[1],
                                              'med_uC': self.unc_c_box[2],'q3_uC': self.unc_c_box[3],
                                              'max_uC': self.unc_c_box[4],'min_uF': self.unc_f_box[0],
                                              'q1_uF': self.unc_f_box[1],'med_uF': self.unc_f_box[2],
                                              'q3_uF': self.unc_f_box[3], 'max_uF': self.unc_f_box[4], 'n_inference': num},
                                             ignore_index=True)


        elif self.model.name == "sr2":


            # X_hr_pred[X_hr_pred < 0] = 0
            # X_hr_pred[X_hr_pred > 1] = 1
            # # compute mean image
            # mean_img = np.mean(X_hr_pred,axis=0)
            # # compute uncertainty
            # std_img = np.std(X_hr_pred,axis=0)
            # compute mean lp
            mean_lp = np.mean(l_pred,axis=0)
            # compute uncertainty
            std_lp = np.std(l_pred,axis=0)
            # evaluate the accuracy of the mean prediction
            self.evaluate_lp(l_test=l_test, l_pred=mean_lp)
            # evalaute the uncertainty of the prediction
            self.evaluate_lp_uncertainty( l_test=l_test,l_pred = mean_lp,std= std_lp)

            # for one image, compute the top3 license plates and store in csv file to provide an example
            #name_top3,lp_top3,max_top3,std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,],lp_std=std_lp[0,],k=3)
            #self.write_dropout(name,lp_top3,max_top3,std_top3,num)
            # Comptue SSIM of the batch
            #self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            # show results
            # self.show_img_lp(X_lr_test, X_hr_test, l_test, mean_img, std_img, name, mean_lp, std_lp, num)
            # store results
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference']== num)])  == 0:

                self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                          'accuracyChar': self.acc_char, 'accuracyLp':self.acc_lp,
                                          'uncertaintyCorrect': self.unc_c,'uncertaintyFalse': self.unc_f,
                                          'AUC_pr': self.auc_pr,'AUC_roc': self.auc_roc, 'n_inference': num, 'SSIM':0},
                                         ignore_index=True)
        self.df.to_csv(self.logs_file,sep=';', index=False)

    def show_results(self,X_lr_test, X_hr_test,l_test,name):

        if self.model.name == "sr":

            X_hr_pred = self.model.predict(x = X_lr_test)
            for i in range(X_hr_pred.shape[0]):
                plt.subplot(1,2,1)
                plt.imshow(X_hr_test[i,:,:,0],cmap="gray")
                plt.title("Ground truth")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(X_hr_pred[i, :, :, 0], cmap="gray")
                plt.title("Prediction")
                plt.axis("off")
                plt.savefig(os.path.join(self.folder, f"{name}_{i}.png"))
                plt.show(block=False)
                plt.pause(3)
                plt.close()

        elif self.model.name == "sr2":

            X_hr_pred, l_pred = self.model.predict(x = X_lr_test)

            for i in range(X_hr_pred.shape[0]):
                plt.subplot(1,2,1)
                plt.imshow(X_hr_test[i,:,:,0],cmap="gray")
                plt.title(f"Ground truth {self.lp_to_str(l_test[i,])}")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(X_hr_pred[i, :, :, 0], cmap="gray")
                plt.title(f"Prediction {self.lp_to_str(l_pred[i,])}")
                plt.axis("off")
                plt.savefig(os.path.join(self.folder, f"{name}_{i}.png"))
                plt.show(block=False)
                plt.pause(3)
                plt.close()


    def show_img(self,X_lr_test,X_hr_test,l_test,mean_img,std_img,name):
        plt.subplot(2, 2, 1)
        plt.imshow(mean_img[0, :, :, 0], cmap="gray")
        plt.title("Mean img")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        img = std_img[0, :, :, 0]
        min_std = np.min(np.min(img, axis=0), axis=0)
        max_std = np.max(np.max(img, axis=0), axis=0)
        img_norm = (img - min_std) / (max_std - min_std)
        plt.imshow(img_norm, cmap="gray")
        plt.title("Std {0:04f} {1:.02f} ".format(min_std, max_std))
        plt.axis("off")  #

        plt.subplot(2, 2, 3)
        plt.imshow(X_lr_test[0, :, :, 0], cmap="gray")
        plt.title("Input")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(X_hr_test[0, :, :, 0], cmap="gray")
        plt.title(f"Ground truth {self.lp_to_str(l_test[0,])}")
        plt.axis("off")
        plt.savefig(os.path.join(self.folder, f"{name}.png"))

        plt.show(block=False)
        #plt.pause(1)
        plt.close()

    # Function to display the mean image along with the mean lp and the top-3 lps
    def show_img_lp(self,X_lr_test,X_hr_test,l_test,mean_img,std_img,name,mean_lp,std_lp,num):
        plt.subplot(2, 2, 1)
        plt.imshow(mean_img[0, :, :, 0], cmap="gray")
        plt.title(f"Mean img {self.lp_to_str(mean_lp[0,])}")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        img = std_img[0, :, :, 0]
        min_std = np.min(np.min(img, axis=0), axis=0)
        max_std = np.max(np.max(img, axis=0), axis=0)
        img_norm = (img - min_std) / (max_std - min_std)
        name_top3, lp_top3, max_top3, std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,], lp_std=std_lp[0,], k=3)
        self.write_dropout(name, lp_top3, max_top3, std_top3, num)

        plt.imshow(img_norm, cmap="gray")
        plt.title("Std {0:04f} {1:.02f} \n{2}".format(min_std, max_std, name_top3))
        plt.axis("off")  #

        plt.subplot(2, 2, 3)
        plt.imshow(X_lr_test[0, :, :, 0], cmap="gray")
        plt.title("Input")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(X_hr_test[0, :, :, 0], cmap="gray")
        plt.title(f"Ground truth {self.lp_to_str(l_test[0,])}")
        plt.axis("off")
        plt.savefig(os.path.join(self.folder, f"{name}.png"))

        plt.show(block=False)
        #plt.pause(1)
        plt.close()

    def show_lp(self,X_lr_test,l_test,name,num):
        plt.imshow(X_lr_test[0, :, :, 0], cmap="gray")
        plt.title(f"Ground truth {self.lp_to_str(l_test[0,])}")
        plt.axis("off")
        plt.savefig(os.path.join(self.folder, f"{name}_{num}_lr.png"))

        plt.show(block=False)
        #plt.pause(1)
        plt.close()


    # Obtaining the current timestamp in an human-readable way
    def timestamp(self):
        timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')

        return timestamp