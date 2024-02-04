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
dir = f"czech//copyDropout"
dirs = os.listdir(os.path.join(job_dir, dir))
name = "copyDropout"
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
n_inference = [1,5,10,20,30,40,50,60,70,80,90,100]
additional50 = False
markers = ["o", "s", "*", "^", "d", "p", "h", "v", "^", "<", ">","h", "d", "p", "h", "v", "^", "<", ">","h"]
std_min = 0.0001
std_max = 0.5
onlyrange = True

results_acc = np.zeros((len(dirs),5,len(n_inference)))
results_auc = np.zeros((len(dirs),5,len(n_inference)))


for i in range(len(noise_all)):

    for m in range(len(metric)):

        tmp_values = []

        if "accuracy" in metric[m]:
            lim = (0.0, 1.01)
        if "uncertainty" in metric[m]:
            lim = (0, 0.6)
        if "AUC" in metric[m]:
            lim = (0.0, 1.01)

        for j, dir_tmp in enumerate(dirs):

            for k in range(len(n_inference)):

                df = eval_data[dir_tmp]

                if onlyrange:
                    tmp_print = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                                   & (df["std"] >= std_min) & (df["std"] <= std_max)]
                else:
                    tmp_print = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])]
                list_tmp = tmp_print[metric[m]].values
                list_tmp = list_tmp[list_tmp >= 0]
                tmp_values.append(np.mean(list_tmp))
                if metric[m] == "accuracyChar":
                    results_acc[j,i, k] = np.mean(list_tmp)
                else:
                    results_auc[j,i, k] = np.mean(list_tmp)

for p in range(len(noise_all)):
    plt.figure(p)
    legend_entries = []
    for j, dir_tmp in enumerate(dirs):
        legend_entries.append(f"{dir_tmp}")
        plt.plot(n_inference,results_acc[j,p,:])
    plt.legend(legend_entries)
    plt.title(f"noise {noise_all[p]} acc")
    tikzplotlib.save(os.path.join(os.getcwd(), "..\\evaluation\\plots",
                                  f"noise-{name}-{noise_all[p]}-acc.tex"))

for p in range(len(noise_all)):
    plt.figure(p)
    legend_entries = []
    for j, dir_tmp in enumerate(dirs):
        legend_entries.append(f"{dir_tmp}")
        plt.plot(n_inference,results_auc[j,p,:])
    plt.legend(legend_entries)
    plt.title(f"noise {noise_all[p]} auc")
    tikzplotlib.save(os.path.join(os.getcwd(), "..\\evaluation\\plots",
                                  f"noise-{name}-{noise_all[p]}-auc.tex"))



for i, dir_tmp in enumerate(dirs):
    if exclude in dir_tmp: continue
    eval_file = os.path.join(os.path.join(job_dir,dir,dir_tmp),'blur.csv')
    eval_data[dir_tmp] = pd.read_csv(eval_file,sep=';')
dirs = [tmp for tmp in dirs if exclude not in tmp]

noise_all = ["horizontal", "vertical"]

std_min = 0
std_max = 10

for i in range(len(noise_all)):

    for m in range(len(metric)):

        tmp_values = []
        legend_entries = []
        if "accuracy" in metric[m]:
            lim = (0.0, 1.01)
        if "uncertainty" in metric[m]:
            lim = (0, 0.6)
        if "AUC" in metric[m]:
            lim = (0.0, 1.01)

        for j, dir_tmp in enumerate(dirs):
            legend_entries.append(f"{dir_tmp}")
            for k in range(len(n_inference)):


                df = eval_data[dir_tmp]

                if onlyrange:
                    tmp_print = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                                   & (df["kernel"] >= std_min) & (df["kernel"] <= std_max)]
                else:
                    tmp_print = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])]
                list_tmp = tmp_print[metric[m]].values
                list_tmp = list_tmp[list_tmp >= 0]
                tmp_values.append(np.mean(list_tmp))

                if metric[m] == "accuracyChar":
                    results_acc[j,i+3, k] = np.mean(list_tmp)
                else:
                    results_auc[j,i*3, k] = np.mean(list_tmp)


for p in range(len(noise_all)):
    plt.figure(p)
    legend_entries = []
    for j, dir_tmp in enumerate(dirs):
        legend_entries.append(f"{dir_tmp}")
        plt.plot(n_inference,results_acc[j,p+3,:])
    plt.legend(legend_entries)
    plt.title(f"blur {noise_all[p]} acc")
    tikzplotlib.save(os.path.join(os.getcwd(), "..\\evaluation\\plots",
                                  f"blur-{name}-{noise_all[p]}-acc.tex"))

for p in range(len(noise_all)):
    plt.figure(p)
    legend_entries = []
    for j, dir_tmp in enumerate(dirs):
        legend_entries.append(f"{dir_tmp}")
        plt.plot(n_inference,results_auc[j,p+3,:])
    plt.legend(legend_entries)
    plt.title(f"blur {noise_all[p]} auc")
    tikzplotlib.save(os.path.join(os.getcwd(), "..\\evaluation\\plots",
                                  f"blur-{name}-{noise_all[p]}-auc.tex"))
