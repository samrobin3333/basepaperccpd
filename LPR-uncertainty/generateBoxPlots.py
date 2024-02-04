import json

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib

def get_boxes(tmp,metric, dir_tmp):
    q3 = tmp[f"q3_{metric}"].values[0]
    q1 = tmp[f"q1_{metric}"].values[0]
    iqr = q3 - q1


    boxes = {
            'label': f"{dir_tmp}-{metric}",
            'whislo': q1 - 1.5 * iqr,  # Bottom whisker position
            'q1': q1,  # First quartile (25th percentile)
            'med': tmp[f"med_{metric}"].values[0],  # Median         (50th percentile)
            'q3': q3,  # Third quartile (75th percentile)
            'whishi': q3 + 1.5 * iqr,  # Top whisker position
            'fliers': [tmp[f"min_{metric}"].values[0], tmp[f"max_{metric}"].values[0]]  # Outliers
        }

    return boxes

job_dir = os.path.abspath("..//checkpoints")

# Test specific folder
dirs = ["dropout1_sr2_w1"]
dir = f"czech"
# Test all folders in the subfolder
dir = f"czech//testBox2"
dirs = os.listdir(os.path.join(job_dir, dir))
name = "d1"
eval_data = {}
#std_all = np.logspace(-3, 0, 10)
fs = 7
rot = 10
exclude = "tmp"

for i, dir_tmp in enumerate(dirs):
    if exclude in dir_tmp: continue
    eval_file = os.path.join(os.path.join(job_dir,dir,dir_tmp),'noise.csv')
    eval_data[dir_tmp] = pd.read_csv(eval_file,sep=';')
dirs = [tmp for tmp in dirs if exclude not in tmp]

noise_all = ["sp", "gaussian", "speckle"]

n_inference = [5]
markers = ["o", "s", "*", "^", "d", "p", "h", "v", "^", "<", ">","h"]
std_cur = 0.2
metric = ["uF","uC"]



additional50 = True

for i in range(len(noise_all)):



    for k in range(len(n_inference)):

        boxes = []
        for j, dir_tmp in enumerate(dirs):

            df = eval_data[dir_tmp]
            tmp = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                     & (df["std"] == std_cur)]

            boxes.append(get_boxes(tmp, "uC", dir_tmp))
            boxes.append(get_boxes(tmp, "uF", dir_tmp))

            if additional50:
                if dir_tmp[0] == 'd':
                    df = eval_data[dir_tmp]

                    tmp = df[(df["noiseType"] == noise_all[i]) & (df["n_inference"] == 50)
                             & (df["std"] == std_cur)]
                    boxes.append(get_boxes(tmp,"uC", f"{dir_tmp}-50"))
                    boxes.append(get_boxes(tmp, "uF", f"{dir_tmp}-50"))
        fig, ax = plt.subplots()
        ax.bxp(boxes, showfliers=False)
        ax.set_ylabel("uncertainty")
        plt.xticks(rotation=rot)
        ax.set_ylim((-0.5, 1.0))
        plt.title(f"{name} {noise_all[i]} boxplots {n_inference[k]}", fontsize=16)
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        #tikzplotlib.save(os.path.join(os.getcwd(), "..\\evaluation\\plots",
        #                              f"noise-{name}-{noise_all[i]}-box-{n_inference[k]}.tex"))
        plt.savefig(os.path.join(os.getcwd(), "..\\evaluation\\plots",
                                 f"noise-{name}-{noise_all[i]}-box-{n_inference[k]}-{std_cur}.png"), dpi=300)





for i, dir_tmp in enumerate(dirs):
    if exclude in dir_tmp: continue
    eval_file = os.path.join(os.path.join(job_dir,dir,dir_tmp),'blur.csv')
    eval_data[dir_tmp] = pd.read_csv(eval_file,sep=';')
dirs = [tmp for tmp in dirs if exclude not in tmp]

noise_all = ["horizontal", "vertical"]

std_cur = 5
std_max = 10


for i in range(len(noise_all)):

    for k in range(len(n_inference)):
        boxes = []
        for j, dir_tmp in enumerate(dirs):

            df = eval_data[dir_tmp]

            tmp = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == n_inference[k])
                         & (df["kernel"] ==std_cur) ]

            boxes.append(get_boxes(tmp, "uC", dir_tmp))
            boxes.append(get_boxes(tmp, "uF", dir_tmp))
            if additional50:
                if dir_tmp[0] == 'd':

                    df = eval_data[dir_tmp]
                    tmp = df[(df["blurType"] == noise_all[i]) & (df["n_inference"] == 50)
                                 &(df["kernel"] ==std_cur)]
                    boxes.append(get_boxes(tmp,"uC", f"{dir_tmp}-50"))
                    boxes.append(get_boxes(tmp, "uF", f"{dir_tmp}-50"))


        fig, ax = plt.subplots()
        ax.bxp(boxes, showfliers=False)
        ax.set_ylabel("uncertainty")
        plt.xticks(rotation=rot)
        plt.title(f"{name} {noise_all[i]} boxplots {n_inference[k]}", fontsize=16)
        ax.tick_params(axis='x', labelsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        ax.set_ylim((-0.5, 1.0))
        #tikzplotlib.save(os.path.join(os.getcwd(),"..\\evaluation\\plots",
        #                              f"blur-{name}-{noise_all[i]}-box-{n_inference[k]}.tex"))
        fig.savefig(os.path.join(os.getcwd(), "..\\evaluation\\plots",
                                 f"blur-{name}-{noise_all[i]}-box-{n_inference[k]}-{std_cur}.png"), dpi=300)
