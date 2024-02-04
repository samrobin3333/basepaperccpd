from utils.datasets import Dataset
from utils.config import get_dicts_test
from utils.tester import Tester
import tensorflow as tf
import keras 
from keras import backend as K
import os
import matplotlib.pyplot as plt

dictLP = ["0","1","2","3","4","5","6","7","8","9"
                       ,"A","B","C","D","E","F","G","H","I","J","K","L","M",
                       "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
                       ,""]



def lp_to_str(lp_pred,dictLP,n_classes = 37):
    str = ""

    for i in range(7):
        str += dictLP[lp_pred[i * n_classes:(i + 1) * n_classes].argmax()]
    return str

data = get_dicts_test()

#create a dataset using the parsed arguments related to data
dataset = Dataset(data=data)
# preload the hdf5 files for faster access
dataset.load_files()
# load the data
test_dropout = False
test_be = True
n_ensemble = 5
n_dropout = 5

X_lr_test, _, l_test = dataset.load_data(mode='test', num=10)



l_test[l_test > 1] = 0
l_test[l_test < 0] = 0

t = lp_to_str(lp_pred=l_test[0],dictLP=dictLP)
job_dir = os.path.abspath("..//checkpoints")

# Test specific folder
#dirs = ["bE_cl"]
#dir = f"{data['dataset']}"
# Test all folders in the subfolder
dir = f"{data['dataset']}//testMixed"
dirs = os.listdir(os.path.join(job_dir,dir))
print(dirs)

#if test_dropout:

result_ensemble = []
result_ensemble_sr2 = []

resultFile = open(f"example_{data['noiseType']}_{data['noise_low']}_{data['blurType']}_{data['blur_strength']}_{data['blur_kernel']}.txt","w")

for i in range(len(dirs)):
    dir_curr = os.path.join(job_dir, dir, dirs[i])
    #print(dir_curr)
    tmp = os.path.split(dir_curr)
    tmp = tmp[1]
    tester = Tester(folder=dir_curr, epoch=-1, data=data)
    tester.load_pretrained()
    tester.create_dataframe()

    plt.imsave(f"example_{data['noiseType']}_{data['noise_low']}_{data['blurType']}_{data['blur_strength']}_{data['blur_kernel']}.png", X_lr_test[0, :, :, 0], cmap='gray')

    if "dropout" in dir_curr:

        res = tester.test_dropout(X_lr_test=X_lr_test, X_hr_test=None, l_test=l_test,
                            name=f"deviation_{data['noiseType']}_{data['noise_low']}_{n_dropout}", num=n_dropout,dataframe=False)


        print(tmp + res)
        resultFile.write(f"{tmp + res} \n")
        del tester
        K.clear_session()
        tester = Tester(folder=dir_curr, epoch=-1, data=data)
        tester.load_pretrained()
        tester.create_dataframe()
        res = tester.test_dropout(X_lr_test=X_lr_test, X_hr_test=None, l_test=l_test,
                            name=f"deviation_{data['noiseType']}_{data['noise_low']}_{30}", num=30,dataframe=False)
        resultFile.write(f"{tmp + res} \n")

    elif "bE" in dir_curr:#

        a = tester.test_be(X_lr_test=X_lr_test, X_hr_test=X_lr_test, l_test=l_test,
                       name=f"deviation_{data['noiseType']}_{data['noise_low']}_{n_ensemble}", dataframe=False)#
        print(a)
        resultFile.write(f"{a} \n")

    elif "ensemble" in dir_curr:
        if "cl" in dir_curr:
            result_ensemble.append(tester.test_ensemble(X_lr_test=X_lr_test))
        else:
            result_ensemble_sr2.append(tester.test_ensemble(X_lr_test=X_lr_test))
    del tester
    K.clear_session()

dir_curr = os.path.join(job_dir, dir, "ensemble_cl_1")
tester = Tester(folder=dir_curr, epoch=-1, data=data)
tester.load_pretrained()
tester.create_dataframe()
c= tester.evaluate_ensemble(l_pred=result_ensemble,l_test=l_test,num=5,dataframe=False)
print(c)
resultFile.write(f"{c} \n")
del tester
K.clear_session()


resultFile.close()
# dir_curr = os.path.join(job_dir, dir, "ensemble_cl_1")
# tester = Tester(folder=dir_curr, epoch=-1, data=data)
# tester.load_pretrained()
# tester.create_dataframe()
# tester.evaluate_ensemble(l_pred=result_ensemble_sr2,l_test=l_test,num=5,dataframe=False)


# elif test_be:
#     for i in range(len(dirs)):
#         dir_curr = os.path.join(job_dir,dir,dirs[i])
#         tester = Tester(folder=dir_curr,epoch=-1,data=data)
#         tester.load_pretrained()
#         tester.test_be(X_lr_test=X_lr_test,X_hr_test=X_hr_test,l_test=l_test,
#                             name=f"deviation_{data['noiseType']}_{data['noise_low']}")
#
#         del tester
#         K.clear_session()
#
#
# else:
#
#     for i in range(len(dirs)):
#         print(dirs[i])
#         dir_curr = os.path.join(job_dir,dir,dirs[i])
#         tester = Tester(folder=dir_curr,epoch=-1,data=data)
#         tester.load_pretrained()
#         tester.test(X_lr_test=X_lr_test,X_hr_test=X_hr_test,l_test=l_test)
#         tester.show_results(X_lr_test=X_lr_test,X_hr_test=X_hr_test,l_test=l_test,
#                             name=f"prediction_{data['noiseType']}_{data['noise_low']}")
#         del tester
#         K.clear_session()