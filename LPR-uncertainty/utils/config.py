from utils.parse import args
import os
from contextlib import redirect_stdout
import json

def get_dicts_test():
    data = {}
    data["dataset"] = args.dataset
    data["scale"] = args.scale
    data["noise"] = args.noise
    data["noiseType"] = args.noiseType
    data["noise_low"] = args.noiseLow
    data["noise_high"] = args.noiseHigh
    data["blur"] = args.blur
    data["blurType"] = args.blurType
    data["blur_strength"] = args.blurStrength
    data["blur_kernel"] = args.blurKernel
    data["local_dir"] = args.dataDir
    data["n_images"] = args.num_images

    # dataset specific information
    if data["dataset"] == "mnist":
        data["in_shape"] = (int(28 / args.scale), int(28 / args.scale))
        data["out_shape"] = (28, 28)
        data["n_channels"] = 1
        data["n_classes"] = 10
    elif data["dataset"] == "svhn":
        data["in_shape"] = (int(32 / args.scale), int(32 / args.scale))
        data["out_shape"] = (32, 32)
        data["n_channels"] = 3
        data["n_classes"] = 10
    elif data["dataset"] == "czech":
        data["in_shape"] = (int(120/ args.scale), int(520/ args.scale))
        data["out_shape"] = (120, 520)
        data["n_channels"] = 1
        data["n_classes"] = 37*7
    elif data["dataset"] == "ccpd":
        data["in_shape"] = (int(30 / args.scale), int(120 / args.scale))
        data["out_shape"] = (30, 120)
        data["n_channels"] = 3
        data["n_classes"] =  68*7

    return data

def get_dicts_train():
    # create dictionary for all information regarding the data!
    data = get_dicts_test()

    # create dictionary for all information regarding the hyperparameter of the network
    parameter = {}
    parameter["reg_strength"] = args.reg_strength
    parameter["n_filter"] = args.num_filters
    parameter["kernel_size"] = 3
    parameter["w_sr"] = args.weight_sr
    parameter["w_cl"] = args.weight_cl
    parameter["dropout"] = args.dropout
    parameter["dropout_rate"] = args.dropout_rate
    parameter["batchEnsemble"] = args.batchEnsemble
    parameter["ensemble_size"] = args.ensemble_size
    parameter["dropoutType"] = args.dropoutType
    parameter["bn_in_cl_sr"] = args.bn_in_cl_sr
    parameter["split_sr_from_cl"] = args.split_sr_from_cl

    # create dictionary for all information regarding the design of the network
    design = {}
    design["model_type"] = args.model
    design["cl_net"] = args.cl_net
    design["sr_net"] = args.sr_net
    design["common"] = args.common_block
    design["split"] = args.split
    design["n_res_blocks"] = args.num_res_blocks

    training = {}
    training["n_batch"] = args.batch_size
    training["lr"] = args.learning_rate
    training["momentum"] = args.momentum
    training["epochs"] = args.epochs
    training["epoch_init"] = args.initial_epoch
    training["lr_steps"] = args.learning_rate_step_size
    training["lr_decay"] = args.learning_rate_decay
    if args.dataset == "czech":
        if args.model == "cl":
            training["monitor"] = 'val_loss'
        elif args.model == "sr2":
            training["monitor"] = 'val_cl_loss'
        else:
            training["monitor"] = 'val_loss'
    if args.dataset == "ccpd":
        if args.model == "cl":
            training["monitor"] = 'val_loss'
        elif args.model == "sr2":
            training["monitor"] = 'val_cl_loss'
        else:
            training["monitor"] = 'val_loss'
    training["patience"] = args.patience
    training["lr_min"] = args.min_lr
    training["shuffle"] = True
    training["gn_rate"] = args.rate
    training["gn_T"] = args.T
    training["gn_alpha"] = args.alpha

    # create dictionary for all storing stuff
    store = {}
    store["pretrained_dir"] = os.path.join(args.job_dir,args.dataset,args.pretrained_model)
    store["period"] = args.period
    store["job_dir"] = args.job_dir
    store["dir"] = os.path.join(args.job_dir,args.dataset,args.name)
    store["best"] = args.save_best_models_only


    total = {}
    total["data"] = data
    total["parameter"] = parameter
    total["design"] = design
    total["store"] = store
    total["training"] = training


    if not os.path.isdir(store["dir"]):
        os.makedirs(store["dir"], exist_ok=True)

    json.dump( total, open( "{0}/dict.json".format(store["dir"]), 'w' ))

    return data,parameter,design,store,training

def create_train_workspace(path,name):
    train_dir = os.path.join(path, name)
    models_dir = os.path.join(train_dir, 'models')
    os.makedirs(train_dir, exist_ok=True)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    return train_dir, models_dir

def write_summary(path, model):
    with open(os.path.join(path, 'summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

def write_args(path, args):
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        for k, v in sorted(args.__dict__.items()):
            f.write('{0}={1}\n'.format(k,v))