from utils.datasetCCPD import DatasetCCPD
from utils.config import get_dicts_test
from utils.tester import Tester
import tensorflow as tf
import keras 
from keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt

data = get_dicts_test()
data["dataset"] = "ccpd"
#create a dataset using the parsed arguments related to data

# load the data
test_dropout = False
test_be = True
n_ensemble = 5
n_dropout = 5

job_dir = os.path.abspath("..//checkpoints")



num_test = 5200#99996
print(f"Num test {num_test} ------------------------------------")
num_patches = 5200
patch_size = int(num_test / num_patches)

print(patch_size)

# Test specific folder
#dirs = ["dropout5_cl"]
#dir = f"{data['dataset']}"
# Test all folders in the subfolder
dir = f"{data['dataset']}//long"
dirs = os.listdir(os.path.join(job_dir,dir))


for i in range(len(dirs)):
    print(dirs[i])
    dir_curr = os.path.join(job_dir,dir,dirs[i])
    tester = Tester(folder=dir_curr,epoch=-1,data=data)
    tester.load_pretrained()
    acc_char = []
    acc_lp = []
    auc_pr = []
    auc_roc = []
    for j in range(num_patches):
        if j%10 == 0: print(j)
        dataset = DatasetCCPD(data=data)
        # preload the hdf5 files for faster access
        dataset.load_files()
        tester.create_dataframe()
        X_hr_test, l_test = dataset.load_data(mode='test', num=patch_size,range=True,start=j*patch_size, end=(j+1)*patch_size)
        if "dropout" in dirs[i]:
            acc_char_tmp, acc_lp_tmp,auc_pr_tmp, auc_roc_tmp = tester.test_dropout(X_lr_test=X_hr_test, X_hr_test=None, l_test=l_test,
                            name=f"patch_{j}", num=5, dataframe=False)
        else:
            acc_char_tmp, acc_lp_tmp = tester.test(X_lr_test=X_hr_test,X_hr_test=None,l_test=l_test)
            auc_pr_tmp = -1
            auc_roc_tmp = -1
        acc_char.append(acc_char_tmp)
        acc_lp.append(acc_lp_tmp)
        auc_pr.append(auc_pr_tmp)
        auc_roc.append(auc_roc_tmp)
    #tester.show_results(X_lr_test=X_hr_test,X_hr_test=X_hr_test,l_test=l_test,
    #                   name=f"prediction_{data['noiseType']}_{data['noise_low']}")

    print("--------------------------------")
    print(dirs[i])
    print(f"Mean acc char: {np.mean(acc_char,axis=0)}")
    print(f"Mean acc lp: {np.mean(acc_lp, axis=0)}")
    print(f"Mean auc_pr: {np.mean(auc_pr,axis=0)}")
    print(f"Mean auc_roc: {np.mean(auc_roc, axis=0)}")
    print("--------------------------------")
    del tester
    K.clear_session()