import json

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib
import csv

job_dir = os.path.abspath("..//checkpoints")

# Test specific folder
dirs = ["dropout1_sr2_w1"]
dir = f"czech"
# Test all folders in the subfolder
dir = f"czech//testEnsemble"
dirs = os.listdir(os.path.join(job_dir, dir))
name = "ensemble"
eval_data = {}
#std_all = np.logspace(-3, 0, 10)

fs = 9
dp = 300
exclude = "tmp"
dirs = [tmp for tmp in dirs if exclude not in tmp]



for i, dir_tmp in enumerate(dirs):
    if exclude in dir_tmp: continue
    eval_file = os.path.join(os.path.join(job_dir,dir,dir_tmp),'noise.csv')
    eval_data[dir_tmp] = pd.read_csv(eval_file,sep=';')



noise_all = ["sp", "gaussian", "speckle"]
metric = ["accuracyChar","AUC_pr"]#,"accuracyLp","AUC_roc","uncertaintyCorrect","uncertaintyFalse"
metricName = ["aC","AUC-pr"]#,"aLP","AUC-roc","uC","uF"
n_inference = [5]
additional50 = False
markers = ["o", "s", "*", "^", "d", "p", "h", "v", "^", "<", ">","h", "d", "p", "h", "v", "^", "<", ">","h"]
std_min = 0.0001
std_max = 0.5
onlyrange = True

results_acc = np.zeros((len(dirs),10))
results_auc = np.zeros((len(dirs),10))
counterRes = 0


for i in range(len(noise_all)):


    print("----------------------------")
    print(noise_all[i])

    print("----------------------------")
    for m in range(len(metric)):
        print("----------------------------")
        print(metric[m])
        tmp_values = []
        print("----------------------------")
        for k in range(len(n_inference)):
            if "accuracy" in metric[m]:
                lim=(0.0, 1.01)
            if "uncertainty" in metric[m]:
                lim=(0, 0.6)
            if "AUC" in metric[m]:
                lim = (0.0, 1.01)

            legend_entries = []
            plt.figure()

            for j, dir_tmp in enumerate(dirs):



                legend_entries.append(f"{dir_tmp}")

                df = eval_data[dir_tmp]
                print("----------------------------")
                print(f"{dir_tmp} with inference {n_inference[k]}")
                if onlyrange:
                    tmp_print = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                                   & (df["std"] >= std_min) & (df["std"] <= std_max)]
                else:
                    tmp_print = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])]
                list_tmp = tmp_print[metric[m]].values
                list_tmp = list_tmp[list_tmp >= 0]
                tmp_values.append(np.mean(list_tmp))
                if metric[m] == "accuracyChar":
                    results_acc[j, counterRes] = np.mean(list_tmp)
                else:
                    results_auc[j, counterRes] = np.mean(list_tmp)


                print(f"Mean {metric[m]}: {np.mean(list_tmp):0.3f}")
                print("----------------------------")
                if "AUC" in metric[m]:
                    tmp = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                             & (df["std"] >= std_min) & (df["std"] <= std_max) & (df[metric[m]] >= 0)]
                else:
                    tmp = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                             & (df["std"] >= std_min) & (df["std"] <= std_max)]


                plt.plot(tmp["std"].values,tmp[metric[m]].values)

                if additional50:
                    if "dropout" in dir_tmp:
                        legend_entries.append(f"{dir_tmp}-50")
                        df = eval_data[dir_tmp]

                        print(f"{dir_tmp} with inference {n_inference[k]}")
                        if onlyrange:
                            tmp_print = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == 50)
                                           & (df["std"] >= std_min) & (df["std"] <= std_max)]
                        else:
                            tmp_print = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == 50)]
                        list_tmp = tmp_print[metric[m]].values
                        list_tmp = list_tmp[list_tmp >= 0]
                        if metric[m] == "accuracyChar":
                            results_acc[j, counterRes + 1] = np.mean(list_tmp)
                        else:
                            results_auc[j, counterRes + 1] = np.mean(list_tmp)
                        print(f"Mean {metric[m]} - 50: {np.mean(list_tmp)}")
                        print("----------------------------")
                        if "AUC" in metric[m]:
                            tmp = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == 50)
                                     & (df["std"] >= std_min) & (df["std"] <= std_max) & (df[metric[m]] >= 0)]
                        else:
                            tmp = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == 50)
                                     & (df["std"] >= std_min) & (df["std"] <= std_max)]
                        plt.plot(tmp["std"].values,tmp[metric[m]].values)


            plt.legend(legend_entries,loc=0,fontsize=fs)
            plt.tick_params(axis='x', labelsize=fs)
            plt.tick_params(axis='y', labelsize=fs)
            plt.xscale('log')

            plt.title(f"{name} {noise_all[i]} {metric[m]} {n_inference[k]}", fontsize=fs)

            tikzplotlib.save(os.path.join(os.getcwd(),"..\\evaluation\\plots",
                                          f"noise-{name}-{noise_all[i]}-{metricName[m]}-{n_inference[k]}.tex"))
            plt.savefig(os.path.join(os.getcwd(),"..\\evaluation\\plots",
                                     f"noise-{name}-{noise_all[i]}-{metricName[m]}-{n_inference[k]}.png"), dpi=dp)

            _, b = zip(*sorted(zip(tmp_values, dirs),reverse=True))
            print(b)

    if additional50:
        counterRes += 2
    else:
        counterRes +=1



for i, dir_tmp in enumerate(dirs):
    if exclude in dir_tmp: continue
    eval_file = os.path.join(os.path.join(job_dir,dir,dir_tmp),'blur.csv')
    eval_data[dir_tmp] = pd.read_csv(eval_file,sep=';')
dirs = [tmp for tmp in dirs if exclude not in tmp]

noise_all = ["horizontal", "vertical"]

std_min = 0
std_max = 10


for i in range(len(noise_all)):
    print("----------------------------")
    print(noise_all[i])
    print("----------------------------")
    for m in range(len(metric)):
        print("----------------------------")
        print(metric[m])
        tmp_values = []
        print("----------------------------")
        for k in range(len(n_inference)):
            if "accuracy" in metric[m]:
                lim=(0.0, 1.01)
            if "uncertainty" in metric[m]:
                lim=(0, 0.5)
            if "AUC" in metric[m]:
                lim = (0.0, 1.01)

            legend_entries = []
            plt.figure()
            for j, dir_tmp in enumerate(dirs):
                if j == 0: continue

                legend_entries.append(f"{dir_tmp}")

                df = eval_data[dir_tmp]
                print("----------------------------")
                print(f"{dir_tmp} with inference {n_inference[k]}")
                if onlyrange:
                    tmp_print = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                                   & (df["kernel"] >= std_min) & (df["kernel"] <= std_max)]
                else:
                    tmp_print = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])]
                list_tmp = tmp_print[metric[m]].values
                list_tmp = list_tmp[list_tmp >= 0]
                tmp_values.append(np.mean(list_tmp))
                print(f"Mean {metric[m]}: {np.mean(list_tmp):0.3f}")
                print("----------------------------")
                if metric[m] == "accuracyChar":
                    results_acc[j, counterRes] = np.mean(list_tmp)
                else:
                    results_auc[j, counterRes] = np.mean(list_tmp)

                if "AUC" in metric[m]:
                    tmp = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                             & (df["kernel"] >= std_min) & (df["kernel"] <= std_max)  & (df[metric[m]] >= 0)]
                else:
                    tmp = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                         & (df["kernel"] >= std_min)  & (df["kernel"] <= std_max)]

                plt.plot(tmp["kernel"].values, tmp[metric[m]].values)

                if additional50:
                    if "dropout" in dir_tmp:
                        legend_entries.append(f"{dir_tmp}-50")
                        df = eval_data[dir_tmp]
                        print("----------------------------")
                        print(f"{dir_tmp} with inference {n_inference[k]}")
                        if onlyrange:
                            tmp_print = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == 50)
                                           & (df["kernel"] >= std_min) & (df["kernel"] <= std_max)]
                        else:
                            tmp_print = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == 50)]
                        list_tmp = tmp_print[metric[m]].values
                        list_tmp = list_tmp[list_tmp >= 0]
                        print(f"Mean {metric[m]} - 50: {np.mean(list_tmp)}")
                        print("----------------------------")

                        if metric[m] == "accuracyChar":
                            results_acc[j, counterRes + 1] = np.mean(list_tmp)
                        else:
                            results_auc[j, counterRes + 1] = np.mean(list_tmp)

                        if "AUC" in metric[m]:
                            tmp = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                                 & (df["kernel"] >= std_min) & (df["kernel"] <= std_max)  & (df[metric[m]] >= 0)]
                        else:
                            tmp = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                                 & (df["kernel"] >= std_min) & (df["kernel"] <= std_max)]
                        plt.plot(tmp["kernel"].values, tmp[metric[m]].values)


            plt.legend(legend_entries,loc=0,fontsize=fs)
            plt.tick_params(axis='x', labelsize=fs)
            plt.tick_params(axis='y', labelsize=fs)
            plt.xscale('log')
            plt.title(f"{name} {noise_all[i]} {metric[m]} {n_inference[k]}", fontsize=fs)
            tikzplotlib.save(os.path.join(os.getcwd(),"..\\evaluation\\plots",
                                          f"blur-{name}-{noise_all[i]}-{metricName[m]}-{n_inference[k]}.tex"))
            plt.savefig(os.path.join(os.getcwd(),"..\\evaluation\\plots",
                                     f"blur-{name}-{noise_all[i]}-{metricName[m]}-{n_inference[k]}.png"), dpi=dp)

            _, b = zip(*sorted(zip(tmp_values, dirs),reverse=True))
            print(b)

    if additional50:
        counterRes += 2
    else:
        counterRes +=1


endLoop = 5
if additional50:
    endLoop = 10

for i, dir_tmp in enumerate(dirs):
    print_str = f"{dir_tmp}"
    for j in range(endLoop):
        print_str += f"& {results_acc[i,j]:0.3f} "
    print_str += "\\"
    print_str += "\\"
    print(print_str)

print("----------------------------------")

for i, dir_tmp in enumerate(dirs):
    print_str = f"{dir_tmp}"
    for j in range(endLoop):
        print_str += f"& {results_auc[i,j]:0.3f} "
    print_str += "\\"
    print_str += "\\"
    print(print_str)