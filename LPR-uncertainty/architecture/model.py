from keras import Input
#import tensorflow as tf
from architecture.networks import common_block_res,common_block_conv,common_fsrcnn,\
    fsrcnn_red,wdsr,fsrcnn,resnet_mine_red,licenseplate_cnn,wdsr_dropout,licenseplate_cnn_dropout,fsrcnn_red_dropout,\
    fsrcnn_dropout,fsrcnn_red_be,wdsr_be,fsrcnn_be,licenseplate_cnn_be
from keras.models import Model

class BuildModel:
    def __init__(self,data,parameter,design):
        self.in_shape = data["in_shape"]
        self.n_channels = data["n_channels"]
        self.n_classes = int(data["n_classes"]/7)
        self.scale = data["scale"]
        self.reg_strength = parameter["reg_strength"]
        self.n_filter = parameter["n_filter"]
        self.kernel_size = parameter["kernel_size"]
        self.cl_net = design["cl_net"]
        self.sr_net = design["sr_net"]
        self.model_type =  design["model_type"]
        self.common = design["common"]
        self.split = design["split"]
        self.n_res_blocks = design["n_res_blocks"]
        self.dropout = parameter["dropout"]


        if ("split_sr_from_cl" not in list(parameter.keys())):
            self.split_sr_from_cl = False
        else:
            self.split_sr_from_cl = parameter["split_sr_from_cl"]

        if ("dropoutType" not in list(parameter.keys())):
            self.dropoutType = "all"
        else:
            self.dropoutType = parameter["dropoutType"]

        if ("bn_in_cl_sr" not in list(parameter.keys())):
            self.bn_in_cl_sr = False
        else:
            self.bn_in_cl_sr = parameter["bn_in_cl_sr"]


        self.dropout_rate = parameter["dropout_rate"]
        self.batchEnsemble = parameter["batchEnsemble"]
        self.ensemble_size = parameter["ensemble_size"]

    def setup_model(self):
        # First the common layer

        if self.model_type != "cl" and self.model_type != "sr"  and self.model_type != "sr2": return

        x_in = Input(shape=(self.in_shape[0], self.in_shape[1], self.n_channels))

        if self.model_type == "cl":
            x = self.setup_common(x_in)
            x_cl = self.setup_cl(x)
            return Model(inputs=x_in, outputs=x_cl, name="cl")

        if self.model_type == "sr":
            x = self.setup_common(x_in)
            x_sr = self.setup_sr(x)
            return Model(inputs=x_in, outputs=x_sr, name="sr")

        x = self.setup_common(x_in)
        x_sr = self.setup_sr(x)
        x_cl = self.setup_cl(x)

        return Model(inputs = x_in, outputs = [x_sr, x_cl], name="sr2")


    def setup_common(self,x):

        if self.common != "res" and self.common != "conv": return x

        if self.common == "res":
            return common_block_res(x,split=self.split,n_filter=self.n_filter, kernel_size=self.kernel_size,
                                    dropout=self.dropout,dropout_rate=self.dropout_rate,
                                    batch_ensemble=self.batchEnsemble, ensemble_size=self.ensemble_size)



        return common_block_conv(x,split=self.split,n_filter=self.n_filter,
                        kernel_size=self.kernel_size,reg_strength=self.reg_strength,
                        dropout=self.dropout,dropout_rate=self.dropout_rate,
                        batch_ensemble=self.batchEnsemble, ensemble_size=self.ensemble_size)


    def setup_sr(self,x):

        if self.sr_net != "fsrcnn" and self.sr_net != "wdsr": return x

        if self.sr_net == "fsrcnn":
            if self.dropout:
                if self.dropoutType == "all":
                    return fsrcnn_dropout(x, scale=self.scale, n_channels=self.n_channels,
                                          reg_strength=self.reg_strength,dropout_rate=self.dropout_rate,
                                          split_sr_from_cl=self.split_sr_from_cl)
                if not self.bn_in_cl_sr:
                    return fsrcnn_dropout(x, scale=self.scale, n_channels=self.n_channels,
                                      reg_strength=self.reg_strength, dropout_rate=0.0)
            if self.batchEnsemble:
                return fsrcnn_be(x, scale=self.scale, n_channels=self.n_channels, reg_strength=self.reg_strength,
                                         ensemble_size=self.ensemble_size, split_sr_from_cl=self.split_sr_from_cl)

            return fsrcnn(x,scale=self.scale,n_channels=self.n_channels, reg_strength=self.reg_strength)


        if self.dropout:
            if self.dropoutType == "all":
                return wdsr_dropout(x, scale=self.scale, n_res_blocks=self.n_res_blocks, n_channels=self.n_channels,
                                n_filter=self.n_filter, reg_strength=self.reg_strength,
                                dropout_rate=self.dropout_rate)
            if not self.bn_in_cl_sr:
                return wdsr_dropout(x, scale=self.scale, n_res_blocks=self.n_res_blocks, n_channels=self.n_channels,
                                n_filter=self.n_filter, reg_strength=self.reg_strength,
                                dropout_rate=0.0)

        if self.batchEnsemble:
            return wdsr_be(x, scale=self.scale, n_res_blocks=self.n_res_blocks, n_channels=self.n_channels,
                           n_filter=self.n_filter, reg_strength=self.reg_strength, ensemble_size=self.ensemble_size)

        return wdsr(x, scale=self.scale, n_res_blocks=self.n_res_blocks, n_channels=self.n_channels,
                    n_filter=self.n_filter, reg_strength=self.reg_strength)


    def setup_cl(self,x):

        if self.cl_net != "res" and self.cl_net != "lp": return x

        if self.cl_net == "res":
            return resnet_mine_red(x,n_classes=self.n_classes,reg_strength=self.reg_strength)

        if self.dropout:
            if self.dropoutType=="all":
                return licenseplate_cnn_dropout(x, reg_strength=self.reg_strength,
                                                dropout_rate=self.dropout_rate,
                                          split_sr_from_cl=self.split_sr_from_cl,split = self.split,
                                                n_classes=self.n_classes)
            if not self.bn_in_cl_sr:
                return licenseplate_cnn_dropout(x, reg_strength=self.reg_strength,
                                            dropout_rate=0.0,n_classes=self.n_classes)
        if self.batchEnsemble:
            return licenseplate_cnn_be(x,reg_strength=self.reg_strength,ensemble_size=self.ensemble_size,
                                       split_sr_from_cl=self.split_sr_from_cl,split = self.split,n_classes=self.n_classes)

        return licenseplate_cnn(x,reg_strength=self.reg_strength,n_classes=self.n_classes)



