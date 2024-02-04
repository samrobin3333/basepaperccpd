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

class Tester:
    def __init__(self,folder,epoch,data):
        self.folder = folder
        self.epoch = epoch
        self.dictLP = ["0","1","2","3","4","5","6","7","8","9"
                       ,"A","B","C","D","E","F","G","H","I","J","K","L","M",
                       "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
                       ,""]

        self.dictLP_ccpd = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣",
                            "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                            "新", "警", "学",'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
                            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

        self.data = data
        if self.data["dataset"] == "czech":
            self.n_classes = 37
        elif self.data["dataset"] == "ccpd":
            self.n_classes = 68


    def create_dataframe(self):
        exits = True
        if self.data["noise"]:
            self.logs_file = os.path.join(self.folder, "noise.csv")
            if not os.path.exists(self.logs_file):
                exits = False
                self.df = pd.DataFrame()
                self.df["noiseType"] = []
                self.df["std"] = []
                self.df["n_inference"] = []

        elif self.data["blur"]:
            self.logs_file = os.path.join(self.folder, "blur.csv")
            if not os.path.exists(self.logs_file):
                exits = False
                self.df = pd.DataFrame()
                self.df["blurType"] = []
                self.df["strength"] = []
                self.df["kernel"] = []
                self.df["n_inference"] = []
        else:
            self.logs_file = os.path.join(self.folder, "normal.csv")
            if not os.path.exists(self.logs_file):
                self.df = pd.DataFrame()
                self.df["n_inference"] = []
                exits = False

        if exits:
            self.df = pd.read_csv(self.logs_file,sep=';')
            return

        if (self.model.name == "cl" or self.model.name == "sr2"):
            self.df["accuracyChar"] = []
            self.df["accuracyLp"] = []
            self.df["uncertaintyCorrect"] = []
            self.df["uncertaintyFalse"] = []
            self.df["AUC_pr"] = []
            self.df["AUC_roc"] = []
            self.df["min_uC"] = []
            self.df["q1_uC"] = []
            self.df["med_uC"] = []
            self.df["q3_uC"] = []
            self.df["max_uC"] = []
            self.df["min_uF"] = []
            self.df["q1_uF"] = []
            self.df["med_uF"] = []
            self.df["q3_uF"] = []
            self.df["max_uF"] = []

        if (self.model.name == "sr" or self.model.name == "sr2"):
            self.df["SSIM"] = []



    def compute_boxplot_variables(self,uncertainty):
        minimum = min(uncertainty)
        maximum = max(uncertainty)
        median = np.percentile(uncertainty,50)
        q1 = np.percentile(uncertainty,25)
        q3 = np.percentile(uncertainty,75)

        return minimum,q1,median,q3,maximum

    def load_pretrained(self):
        # load the information regarding the network from the json file
        total = json.load(open( os.path.join(self.folder,"dict.json"), 'r' ))
        self.store = total["store"]
        self.cl_net = total["design"]["cl_net"]
        total["design"]["model_type"] = "cl"
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

    def batch_ssim(self,X_hr_test, X_hr_pred):
        assert (X_hr_pred.shape == X_hr_test.shape)
        ssim_all = []
        for idx in range(X_hr_pred.shape[0]):
            if X_hr_pred.shape[-1] == 3:
                s = ssim(X_hr_pred[idx], X_hr_test[idx], multichannel=True)
                ssim_all.append(s)

            else:
                s = ssim(X_hr_pred[idx,:,:,0], X_hr_test[idx,:,:,0])
                ssim_all.append(s)

        ssim_all_m = mean(ssim_all)
        ssim_all_s = stdev(ssim_all)
        #print(f"SSIM {ssim_all_m}")
        #self.file.write(f"SSIM {ssim_all_m} \n")

        self.ssim_all_m = ssim_all_m
        self.ssim_all_s = ssim_all_s

    def lp_to_str(self,lp_pred):
        str = ""

        for i in range(7):
            str += self.dictLP[lp_pred[ i *  self.n_classes:(i + 1) *  self.n_classes].argmax()]
        return str
    def lp_to_str_ccpd(self,lp_pred):
        str = ""

        for i in range(7):
            str += self.dictLP_ccpd[lp_pred[ i *  self.n_classes:(i + 1) *  self.n_classes].argmax()]
        return str


    def lp_to_str_topk(self,lp_pred,lp_std,k):
        # store the top-k license plates in case you want to print them in the format "k=j lp"
        name_lp = ""
        name_max = ""
        # Store the top-k license plates
        str_all = []
        #store the top-k max values of the prediction vector lp_pred
        max_all =np.zeros((k,7))
        # store the standard dev at the positions of the top-k max values
        std_all = np.zeros((k,7))
        for j in range(k):
            str = ""
            max = ""
            for i in range(7):
                max_curr =  np.max(lp_pred[ i *  self.n_classes:(i + 1) *  self.n_classes])
                pos_max = np.argmax(lp_pred[ i *  self.n_classes:(i + 1) *  self.n_classes])
                max += "{0:.03f} ".format(max_curr)
                max_all[j,i] = max_curr
                std_all[j,i] = lp_std[pos_max]
                lp_pred[i *  self.n_classes:(i + 1) *  self.n_classes][pos_max] = 0
                str += self.dictLP[pos_max]
            name_lp +=f"k={j} {str}\n"
            name_max += f"{max}\n"
            str_all.append(str)
        name = name_lp
        return name,str_all,max_all,std_all

    def evaluate_lp(self,l_test,l_pred):
        # Character accuracy
        acc_char = 0
        # Accuracy of correct license plates
        acc_lp_all = np.zeros((l_test.shape[0],7))

        for i in range(7):
                # compute accuracy of batch at position i
                acc = accuracy_score(y_true=(l_test[:, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax(axis=1),
                               y_pred=(l_pred[:, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax(axis=1))
                # store accuracies
                acc_lp_all[:,i] = np.where((l_test[:, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax(axis=1)
                                           == (l_pred[:, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax(axis=1),1,0)
                # update character accuracy
                acc_char += acc
        # normalize character accuracy
        acc_char /= 7

        # compute mean accuracy of each batch
        acc_lp_batch = np.mean(acc_lp_all,axis=-1)
        # If accuracy is below 1 -> lp is not correct set to 0
        acc_lp_batch[acc_lp_batch < 1] = 0
        # Compute mean
        acc_lp = np.mean(acc_lp_batch)

        # print accuracies
        #print(f"accuracy char correct {acc_char}")
        #print(f"accuracy whole lp correct {acc_lp}")

        # store accuracies
        self.acc_char = acc_char
        self.acc_lp = acc_lp

    def evaluate_lp_uncertainty(self,l_test,l_pred,std):
        # uncertainty for correctly classified characters
        uncertainty_correct = []
       
        # uncertainty for falsely classified characters
        uncertainty_false = []



      

        for i in range(7):
            for j in range(l_test.shape[0]):
                # if the prediction is correct, store the uncertainty of the correct predictions
                
                tmp = std[j, i *  self.n_classes + (l_pred[j, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax()]

                
                if (l_test[j, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax() == \
                   (l_pred[j, i *  self.n_classes:(i + 1) *  self.n_classes]).argmax():
                    uncertainty_correct.append(tmp)

                    continue
                # if the prediction is false, store the uncertainty of the false predictions
                uncertainty_false.append(tmp)



        if len(uncertainty_false) != 0 and len(uncertainty_correct) != 0:
            num_correct = len(uncertainty_correct)
            num_false = len(uncertainty_false)

            if num_correct > num_false:
                uncertainty_correct_sub = random.sample(uncertainty_correct, num_false)
                label_correct = [0 for i in range(num_false)]
                label_false = [1 for i in range(num_false)]
                uncertainty = uncertainty_correct_sub + uncertainty_false
                label = label_correct + label_false
            else:
                uncertainty_false_sub = random.sample(uncertainty_false, num_correct)
                label_correct = [0 for i in range(num_correct)]
                label_false = [1 for i in range(num_correct)]
                uncertainty = uncertainty_correct + uncertainty_false_sub
                label = label_correct + label_false
            
        if len(uncertainty_false) == 0:
            uncertainty_false = 0
            min_uF, q1_uF, med_uF, q3_uF, max_uF = 0,0,0,0,0
        else:
            min_uF,q1_uF,med_uF,q3_uF,max_uF = self.compute_boxplot_variables(uncertainty_false)

        if len(uncertainty_correct) == 0:
            uncertainty_correct = 0
            min_uC, q1_uC, med_uC, q3_uC, max_uC = 0, 0, 0, 0, 0
        else:
            min_uC, q1_uC, med_uC, q3_uC, max_uC = self.compute_boxplot_variables(uncertainty_correct)

        # print uncertainties
        #print(f"uncertainty of correct characters {np.mean(uncertainty_correct)}")
        #print(f"uncertainty of false characters {np.mean(uncertainty_false)}")
        #self.file.write(f"accuracy {acc_w} \n")
        a_roc = -1
        a_pr = -1
        if type(uncertainty_false) is list:
            fpr, tpr, _ = roc_curve(np.array(label), np.array(uncertainty), pos_label=1)
            pre,rec,_ = precision_recall_curve(y_true=np.array(label), probas_pred=np.array(uncertainty), pos_label=1)
            a_roc = auc(fpr,tpr)
            a_pr = auc(rec, pre)

        #print(f"AUC roc {a_roc}")
        #print(f"AUC pr {a_pr}")

        # Store the uncertainty measures
        self.unc_c = np.mean(uncertainty_correct)
        self.unc_c_box = [min_uC, q1_uC, med_uC, q3_uC, max_uC]
        self.unc_f = np.mean(uncertainty_false)
        self.unc_f_box = [min_uF, q1_uF, med_uF, q3_uF, max_uF]
        self.auc_roc = a_roc
        self.auc_pr = a_pr

    def acc_from_conf(self,matrix):

        return (matrix[0, 0] + matrix[1, 1]) / (matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])

    def evaluate_cl(self,l_test,l_pred):

        matrix_w = confusion_matrix(l_test, l_pred)

        matrix_individual = multilabel_confusion_matrix(l_test, l_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc_individual = []

        for idx in range(10):
            acc_individual.append(self.acc_from_conf(matrix_individual[idx]))

        #self.file.write(f"\nAccuracy indivudual: {acc_individual} \n \n ")

        acc_char = accuracy_score(l_test, l_pred)
        print(f"accuracy {acc_char}")
        #self.file.write(f"accuracy {acc_w} \n")
        f1_w = f1_score(l_test, l_pred, average='weighted')
        #print(f"f1 {f1_w}")
        #self.file.write("f1 {f1_w} \n")

        ## store confusion matrix of worst data in csv file
        np.savetxt(os.path.join(self.store["dir"], "confusion_matrix_all.csv"), matrix_w, fmt='%5i', delimiter=';',
                   newline='\n')

        self.acc_char = acc_char
        self.f1_w = f1_w

    def write_dropout(self,name,lp,max,lp_std,num):
        # name: name of the file where we store the results
        # lp_max: max values in the prediction vectors at the 7 positions of the license plate
        # lp_std: standard deviation at the max positions

        with open(os.path.join(self.folder,f"{name}_{num}.csv"), 'w', newline='\n') as csvfile:
            # header of the csv file
            fieldnames = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                          'P1', 'S1', 'P2', 'S2', 'P3', 'S3', 'P4', 'S4', 'P5', 'S5', 'P6', 'S6', 'P7', 'S7']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=';')

            writer.writeheader()
            for i in range(max.shape[0]):
                dict_curr = {}
                for j in range(7):
                    # Write the character that is predicted
                    dict_curr[f"C{j + 1}"] = lp[i][j]
                    #Write the softmax output
                    dict_curr[f"P{j + 1}"] = max[i,j].astype(np.float16)
                    # Write the uncertainty
                    dict_curr[f"S{j + 1}"] = lp_std[i, j].astype(np.float16)

                writer.writerow(dict_curr)

    def test_be(self,X_lr_test, X_hr_test,l_test,name, dataframe=True):

        # repeat the input by the number of ensembles

        num_images = X_lr_test.shape[0]


        if self.model.name == "cl":
            l_pred_split = []
            #Predictions of the ensemble
            for i in range(0,num_images,10):

                l_pred = self.model.predict(x = np.tile(X_lr_test[i:i+10], [self.n_ensemble, 1, 1, 1]),batch_size=10)
                # split predictions into the individual ensemble member
                l_pred_split.append(np.array_split(l_pred,self.n_ensemble,axis=0))

            l_pred_split = np.concatenate(l_pred_split, axis=1)
            # compute mean lp
            mean_lp = np.mean(l_pred_split,axis=0)
            # compute uncertainty measure
            std_lp = np.std(l_pred_split,axis=0)

            #evaluate the accuracy of the mean prediction
            self.evaluate_lp(l_test=l_test, l_pred=mean_lp)
            # evalaute the uncertainty of the prediction
            self.evaluate_lp_uncertainty( l_test=l_test,l_pred = mean_lp,std= std_lp)

            # for one image, compute the top3 license plates and store in csv file to provide an example
            #name_top3,lp_top3,max_top3,std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,],lp_std=std_lp[0,],k=3)
            #self.write_dropout(name,lp_top3,max_top3,std_top3,self.n_ensemble)
            # Show the results
            #self.show_lp(X_lr_test, l_test, name, self.n_ensemble)

            if dataframe:

                # Store the results
                if self.data["noise"]:
                    if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                                   (self.df['std'] == self.data["noise_low"]) &
                                   (self.df['n_inference'] == self.n_ensemble)]) == 0:
                        self.df = self.df.append({'noiseType': self.data["noiseType"], 'std': self.data["noise_low"],
                                                  'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                                  'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                                  'AUC_pr': self.auc_pr,'AUC_roc': self.auc_roc,'min_uC': self.unc_c_box[0],
                                                  'q1_uC': self.unc_c_box[1],'med_uC': self.unc_c_box[2],
                                                  'q3_uC': self.unc_c_box[3],'max_uC': self.unc_c_box[4],
                                                  'min_uF': self.unc_f_box[0],'q1_uF': self.unc_f_box[1],
                                                  'med_uF': self.unc_f_box[2],'q3_uF': self.unc_f_box[3],
                                                  'max_uF': self.unc_f_box[4],
                                                  'n_inference': self.n_ensemble},
                                                 ignore_index=True)
                elif self.data["blur"]:
                    if len(self.df[(self.df["blurType"] == self.data["blurType"]) &
                                   (self.df['strength'] == self.data["blur_strength"]) &
                                   (self.df['kernel'] == self.data["blur_kernel"]) &
                                   (self.df['n_inference'] == self.n_ensemble)]) == 0:
                        self.df = self.df.append({'blurType': self.data["blurType"], 'strength': self.data["blur_strength"],
                                                  'kernel': self.data["blur_kernel"],
                                                  'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                                  'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                                  'AUC_pr': self.auc_pr,'AUC_roc': self.auc_roc,'min_uC': self.unc_c_box[0],
                                                  'q1_uC': self.unc_c_box[1],'med_uC': self.unc_c_box[2],
                                                  'q3_uC': self.unc_c_box[3],'max_uC': self.unc_c_box[4],
                                                  'min_uF': self.unc_f_box[0],'q1_uF': self.unc_f_box[1],
                                                  'med_uF': self.unc_f_box[2],'q3_uF': self.unc_f_box[3],
                                                  'max_uF': self.unc_f_box[4],
                                                  'n_inference': self.n_ensemble},
                                                 ignore_index=True)


                else:
                    if len( self.df[(self.df['n_inference'] == self.n_ensemble)]) == 0:
                        self.df = self.df.append({'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                                  'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                                  'AUC_pr': self.auc_pr, 'AUC_roc': self.auc_roc,
                                                  'min_uC': self.unc_c_box[0],
                                                  'q1_uC': self.unc_c_box[1], 'med_uC': self.unc_c_box[2],
                                                  'q3_uC': self.unc_c_box[3], 'max_uC': self.unc_c_box[4],
                                                  'min_uF': self.unc_f_box[0], 'q1_uF': self.unc_f_box[1],
                                                  'med_uF': self.unc_f_box[2], 'q3_uF': self.unc_f_box[3],
                                                  'max_uF': self.unc_f_box[4],
                                                  'n_inference': self.n_ensemble},
                                                 ignore_index=True)

            else:

                return f"BatchEnsemble: Acc {self.acc_char:.3f} AccLP {self.acc_lp:.3f} uncertainty correct {self.unc_c:.3f} uncertainty false {self.unc_f:.3f} AUC pr {self.auc_pr:.3f} AUC_roc {self.auc_roc:.3f}"

        elif self.model.name == "sr":
            X_hr_pred_split = []
            # Predictions of the ensemble
            for i in range(0, num_images, 10):
                X_hr_pred = self.model.predict(x=np.tile(X_lr_test[i:i + 10], [self.n_ensemble, 1, 1, 1]), batch_size=10)
                # split predictions into the individual ensemble member
                X_hr_pred_split.append(np.array_split(X_hr_pred, self.n_ensemble, axis=0))

            X_hr_pred = np.concatenate(X_hr_pred_split, axis=1)

            # Predictions of the ensemble
            X_hr_pred = X_hr_pred.astype(np.float32)
            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            # split predictions into the individual ensemble member
            X_hr_pred_split = np.array_split(X_hr_pred, self.n_ensemble, axis=0)
            # compute mean image
            mean_img = np.mean(X_hr_pred_split,axis=0)
            # compute uncertainty
            std_img = np.std(X_hr_pred_split,axis=0)
            # Comptue SSIM of the batch
            self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            # Show the results
            self.show_img(X_lr_test, X_hr_test, l_test, mean_img, std_img,name)
            # Store the results
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference'] == self.n_ensemble)]) == 0:
                self.df = self.df.append({'noiseType': self.data["noiseType"], 'std': self.data["noise_low"],
                                          'n_inference': self.n_ensemble, 'SSIM': self.ssim_all_m},
                                         ignore_index=True)

        else:
            # Predictions of the ensemble

            X_hr_pred_split = []
            l_pred_split = []
            # Predictions of the ensemble
            for i in range(0, num_images, 10):
                X_hr_pred,l_pred = self.model.predict(x=np.tile(X_lr_test[i:i + 10], [self.n_ensemble, 1, 1, 1]), batch_size=10)
                # split predictions into the individual ensemble member
                X_hr_pred_split.append(np.array_split(X_hr_pred, self.n_ensemble, axis=0))
                l_pred_split.append(np.array_split(l_pred, self.n_ensemble, axis=0))

            X_hr_pred_split = np.concatenate(X_hr_pred_split, axis=1)
            l_pred_split = np.concatenate(l_pred_split, axis=1)

            X_hr_pred_split = X_hr_pred_split.astype(np.float32)
            X_hr_pred_split[X_hr_pred_split < 0] = 0
            X_hr_pred_split[X_hr_pred_split > 1] = 1
            # compute mean image
            mean_img = np.mean(X_hr_pred_split,axis=0)
            # compute uncertainty
            std_img = np.std(X_hr_pred_split,axis=0)
            # compute mean lp
            mean_lp = np.mean(l_pred_split,axis=0)
            # compute uncertainty
            std_lp = np.std(l_pred_split,axis=0)
            # evaluate the accuracy of the mean prediction
            self.evaluate_lp(l_test=l_test, l_pred=mean_lp)
            # evalaute the uncertainty of the prediction
            self.evaluate_lp_uncertainty( l_test=l_test,l_pred = mean_lp,std= std_lp)
            # for one image, compute the top3 license plates and store in csv file to provide an example
            #name_top3,lp_top3,max_top3,std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,],lp_std=std_lp[0,],k=3)
            #self.write_dropout(name,lp_top3,max_top3,std_top3,self.n_ensemble)

            # Comptue SSIM of the batch
            self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            # show results
            #self.show_img_lp(X_lr_test, X_hr_test, l_test, mean_img, std_img, name, mean_lp, std_lp, self.n_ensemble)
            # store results

            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference'] == self.n_ensemble)]) == 0:
                self.df = self.df.append({'noiseType': self.data["noiseType"], 'std': self.data["noise_low"],
                                          'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                          'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                          'AUC_pr': self.auc_pr,'AUC_roc': self.auc_roc,
                                          'n_inference': self.n_ensemble, 'SSIM': 0},
                                         ignore_index=True)
        self.df.to_csv(self.logs_file, sep=';', index=False)

    def test_dropout(self,X_lr_test, X_hr_test,l_test,name,num,dataframe=True):

        if self.model.name == "cl":

            l_pred = np.array([ self.model.predict(x = X_lr_test).astype(np.float32) for i in range(num)])

            # compute mean lp
            mean_lp = np.mean(l_pred,axis=0)
            # compute uncertainty measure
            std_lp = np.std(l_pred,axis=0)

            #evaluate the accuracy of the mean prediction
            self.evaluate_lp(l_test=l_test, l_pred=mean_lp)
            # evalaute the uncertainty of the prediction
            self.evaluate_lp_uncertainty( l_test=l_test,l_pred = mean_lp,std= std_lp)

            if self.acc_lp != 1:
                print(f"Model not correct: {name}")
                plt.imsave(f"{name}.png", X_lr_test[0, :, :, :])
                with open(f"{name}.txt","w",encoding="utf-8") as f:
                    for i in range(num):
                        tmp = self.lp_to_str_ccpd(l_pred[i,0,:])

                        f.write(f"{tmp} \n")




            # for one image, compute the top3 license plates and store in csv file to provide an example
            #name_top3,lp_top3,max_top3,std_top3 = self.lp_to_str_topk(lp_pred=mean_lp[0,],lp_std=std_lp[0,],k=3)
            #self.write_dropout(name,lp_top3,max_top3,std_top3,num)
            # Show the results
            # self.show_lp(X_lr_test, l_test, name, num)
            # Store the results

            # make sure the row does not exist
            if dataframe:
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
                elif self.data["blur"]:
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
                                                  'q3_uF': self.unc_f_box[3], 'max_uF': self.unc_f_box[4],'n_inference': num},
                                                 ignore_index=True)
                else:

                    if len(self.df[(self.df['n_inference'] == num)]) == 0:
                        self.df = self.df.append({'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp,
                                                  'uncertaintyCorrect': self.unc_c, 'uncertaintyFalse': self.unc_f,
                                                  'AUC_pr': self.auc_pr, 'AUC_roc': self.auc_roc,
                                                  'min_uC': self.unc_c_box[0],'q1_uC': self.unc_c_box[1],
                                                  'med_uC': self.unc_c_box[2],'q3_uC': self.unc_c_box[3],
                                                  'max_uC': self.unc_c_box[4],'min_uF': self.unc_f_box[0],
                                                  'q1_uF': self.unc_f_box[1],'med_uF': self.unc_f_box[2],
                                                  'q3_uF': self.unc_f_box[3], 'max_uF': self.unc_f_box[4],
                                                  'n_inference': num},
                                                 ignore_index=True)
            else:
                pass
                #return f" {num}: Acc {self.acc_char:.3f} AccLP {self.acc_lp:.3f} uncertainty correct {self.unc_c:.3f} uncertainty false {self.unc_f:.3f} AUC pr {self.auc_pr:.3f} AUC_roc {self.auc_roc:.3f}"

        elif self.model.name == "sr":

            X_hr_pred = [self.model.predict(x = X_lr_test).astype(np.float32) for i in range(num)]

            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            mean_img = np.mean(X_hr_pred,axis=0)
            # compute uncertainty
            std_img = np.std(X_hr_pred,axis=0)
            # Comptue SSIM of the batch
            self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
            # Show the results
            # self.show_img(X_lr_test, X_hr_test, l_test, mean_img, std_img,name)
            # Store the results
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference']== num)])  == 0:

                self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                          'n_inference': num, 'SSIM':self.ssim_all_m}, ignore_index=True)

        else:

            X_hr_pred = np.zeros((num,X_hr_test.shape[0],X_hr_test.shape[1],X_hr_test.shape[2],X_hr_test.shape[3]),dtype=np.float32)
            l_pred = np.zeros((num,l_test.shape[0],l_test.shape[1]),dtype=np.float32)
            for i in range(num):
                X_hr_pred[i,],l_pred[i,] = self.model.predict(x = X_lr_test)


            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            # compute mean image
            mean_img = np.mean(X_hr_pred,axis=0)
            # compute uncertainty
            std_img = np.std(X_hr_pred,axis=0)
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
            self.batch_ssim(X_hr_test=X_hr_test, X_hr_pred=mean_img)
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
        return self.acc_char, self.acc_lp, self.auc_pr, self.auc_roc

    def test(self,X_lr_test, X_hr_test,l_test):

        if self.model.name == "cl":

            l_pred = self.model.predict(x = X_lr_test)


            if self.cl_net == "lp":
                self.evaluate_lp(l_test=l_test,l_pred=l_pred)
            else:
                self.evaluate_cl(l_test=l_test.argmax(axis=1), l_pred=l_pred.argmax(axis=1))

            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference']== 50)])  == 0:

                self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                          'accuracyChar': self.acc_char, 'accuracyLp':self.acc_lp}, ignore_index=True)


        elif self.model.name == "sr":

            X_hr_pred = self.model.predict(x = X_lr_test)
            X_hr_pred = X_hr_pred.astype(np.float32)
            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            self.batch_ssim(X_hr_test=X_hr_test,X_hr_pred=X_hr_pred)
            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference']== 50)])  == 0:

                self.df = self.df.append({'noiseType': self.data["noiseType"],'std': self.data["noise_low"],
                                           'SSIM':self.ssim_all_m}, ignore_index=True)

        else:

            X_hr_pred, l_pred = self.model.predict(x = X_lr_test)
            X_hr_pred = X_hr_pred.astype(np.float32)
            X_hr_pred[X_hr_pred < 0] = 0
            X_hr_pred[X_hr_pred > 1] = 1
            self.batch_ssim(X_hr_test=X_hr_test,X_hr_pred=X_hr_pred)
            if self.cl_net == "lp":
                self.evaluate_lp(l_test=l_test,l_pred=l_pred)
            else:
                self.evaluate_cl(l_test=l_test.argmax(axis=1), l_pred=l_pred.argmax(axis=1))

            if len(self.df[(self.df["noiseType"] == self.data["noiseType"]) &
                           (self.df['std'] == self.data["noise_low"]) &
                           (self.df['n_inference'] == 50)]) == 0:
                self.df = self.df.append({'noiseType': self.data["noiseType"], 'std': self.data["noise_low"],
                                          'accuracyChar': self.acc_char, 'accuracyLp': self.acc_lp},
                                         ignore_index=True)
        #self.df.to_csv(self.logs_file, sep=';', index=False)

        return self.acc_char, self.acc_lp


    def test_ensemble(self,X_lr_test):
        if self.model.name == "cl":

            l_pred = self.model.predict(x=X_lr_test)


        elif self.model.name == "sr":

            X_hr_pred = self.model.predict(x=X_lr_test)

        else:

            X_hr_pred, l_pred = self.model.predict(x=X_lr_test)

        return l_pred

    def evaluate_ensemble(self,l_pred,l_test,num,dataframe=True):

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
            if dataframe:
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
            else:
                return f"Ensemble: Acc {self.acc_char:.3f} AccLP {self.acc_lp:.3f} uncertainty correct {self.unc_c:.3f} uncertainty false {self.unc_f:.3f} AUC pr {self.auc_pr:.3f} AUC_roc {self.auc_roc:.3f}"

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