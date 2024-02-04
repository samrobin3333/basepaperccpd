import argparse

parser = argparse.ArgumentParser(description='Multi-task learning on Super-Resolution and Number classification')

# Add Arguments specific to models, training, testing and so on

# is you want to run the code on another computer and debug on your own
parser.add_argument(
    '--remoteDebug',
    action='store_true',
    help='If set, debug mode in localhost is possible'
)

## Architecture
parser.add_argument(
    '--model',
    help='Model name',
    default='sr2',
    choices=['cl', 'sr2', 'sr'],
    type=str)
parser.add_argument(
    '--sr-net',
    help='network to choose for the super-resolution part',
    choices = ['wdsr','fsrcnn'],
    default='wdsr',
    type=str
)
parser.add_argument(
    '--cl-net',
    help='network to choose for the classification part',
    choices = ['res','lp'],
    default='lp',
    type=str
)
parser.add_argument(
    '--split',
    help='number of res blocks after which we split WDSR',
    default=4,
    type=int
)
parser.add_argument(
    '--num-filters',
    type=int,
    default=32,
    help='number of filters in the conv layers of the residual blocks')
parser.add_argument(
    '--num-res-blocks',
    type=int,
    default=16,
    help='number of residual blocks')
parser.add_argument(
    '--reg-strength',
    type=float,
    default= 0.0,
    help='L2 regularization of kernel weights')
parser.add_argument(
    '--common-block',
    type=str,
    default='res',
    choices = ['res', 'conv'],
    help='which basic block should be used'
)
parser.add_argument(
    '--split-sr-from-cl',
    help='If set, we use the feature extraction from the cl for the sr network',
    action='store_true')


## Training
parser.add_argument(
    '--freeze',
    help='freeze certain weights related to sr, cl, common',
    type=str,
    choices=['sr','None','cl','common'],
    default='None')

parser.add_argument(
    '--batch-size',
    help='Batch size for training steps',
    type=int,
    default=16)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--learning-rate',
    type=float,
    default=0.001,
    help='learning rate')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='learning rate')

# for the learning rate callback
parser.add_argument(
    '--learning-rate-step-size',
    type=int,
    default=30,
    help='learning rate step size in epochs -> learning rate decay at the end')
parser.add_argument(
    '--learning-rate-decay',
    type=float,
    default=0.2,
    help='learning rate decay for reduceLROnPlateu')
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    help='when validation loss increases, after patience epochs the learning rate is divided by 10')
parser.add_argument(
    '--min-lr',
    type=float,
    default=0.0000001,
    help='lr does not go beyond min-lr')
parser.add_argument(
    '--weight-check',
    help='Store the weights for 1 epoch to check if the learning rate choice is ok ',
    action='store_true')
parser.add_argument(
    '--pretrained-model',
    type=str,
    help='path to pre-trained model',
    default = "")
parser.add_argument(
    '--dropoutType',
    type=str,
    help='where to do dropout',
    choices = ['all', 'common'],
    default ="all")
parser.add_argument(
    '--dropout',
    help='Decide if we apply dropout ',
    action='store_true')
parser.add_argument(
    '--bn-in-cl-sr',
    help='If dropouttype is common we can either use bn for cl and sr task or nothing',
    action='store_true')
parser.add_argument(
    '--dropout-rate',
    type=float,
    default=0.5,
    help='dropout rate for each layer')
parser.add_argument(
    '--batchEnsemble',
    help='Decide if we apply BatchEnsemble ',
    action='store_true')
parser.add_argument(
    '--ensemble-size',
    type=int,
    default=5,
    help='Number of ensembles for batch ensemble')


## Arguments for the datasets
parser.add_argument(
    '--dataset',
    help='decide on which dataset you want to train/test you network',
    type=str,
    choices = ['mnist', 'stl','svhn','czech','ccpd'],
    default='czech')
parser.add_argument(
    '--dataDir',
    help='where is your data stored',
    type=str,
    default='../data')
parser.add_argument(
    '--num-images',
    help = 'number of images we use for training',
    type=int,
    default=604388)
parser.add_argument(
    '--scale',
    help='Magnification factor for image super-resolution',
    default=8,
    type=int)
parser.add_argument(
    '--noise',
    help='Add noise to training/validation data',
    action='store_true')
parser.add_argument(
    '--noiseType',
    help = 'type of noise',
    type=str,
    choices=['gaussian', 'sp', 'speckle'],
    default="gaussian")
parser.add_argument(
    '--noiseLow',
    help='Lower bound for the noise',
    type=float,
    default=0.0001)
parser.add_argument(
    '--noiseHigh',
    help='Upper bound for the noise',
    type=float,
    default=0.1)
parser.add_argument(
    '--blur',
    help='Add blur to training/validation data',
    action='store_true')
parser.add_argument(
    '--blurType',
    help = 'type of blur',
    type=str,
    choices=['horizontal', 'vertical', 'defocus'],
    default="gaussian")
parser.add_argument(
    '--blurStrength',
    help='Lower bound for the blur',
    type=float,
    default=0.0001)
parser.add_argument(
    '--blurKernel',
    help='Size of vertical/horizontal blur',
    type=int,
    default=4)




## Multi-task stuff
parser.add_argument(
    '--weight-sr',
    help='weight of the sr_loss in the total loss',
    default=0.1,
    type=float
)
parser.add_argument(
    '--weight-cl',
    help='weight of the classification_loss in the total loss',
    default=0.9,
    type=float
)
# Grad Norm
parser.add_argument(
    '--rate',
    help='learing rate of loss weight updates',
    default=0.01,
    type=float
)
parser.add_argument(
    '--T',
    help='sum of all loss weights',
    default=2,
    type=float
)
parser.add_argument(
    '--alpha',
    help='strength of restoring force',
    default=0.12,
    type=float
)
parser.add_argument(
    '--gradnorm',
    help='Gradorm ',
    action='store_true')
parser.add_argument(
    '--mgda',
    help='Multi-objective optimization',
    action='store_true')


## Storing
parser.add_argument(
    '--period',
    help='the model will be stored after every period-th epoch',
    default= 1,
    type=int
)
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    default= '../checkpoints',
    required=False)
parser.add_argument(
    '--name',
    default='test_stuff',
    help='name of the folder where data is stored')

parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO',
    help='Set logging verbosity')
parser.add_argument(
    '--save-best-models-only',
    action='store_true',
    help='save only models with improved validation psnr (overridden by --benchmark)'
)
parser.add_argument(
    '--save-models-after-epoch',
    type=int,
    default=0,
    help='start saving models only after given epoch')
parser.add_argument(
    '--initial-epoch',
    type=int,
    default=0,
    help='resumes training of provided model if greater than 0')



# WDSR
parser.add_argument(
    '-o',
    '--outdir',
    help='output directory',
    default='./output',
    type=str
    )

parser.add_argument(
    '--random-seed',
    help='Random seed for TensorFlow',
    default=None,
    type=int)


args = parser.parse_args()
if args.dropout or args.batchEnsemble:
    assert (args.dropout != args.batchEnsemble), "Dropout and BatchEnsemble not allowed at the same time"

assert (args.split >= 0), "Splits must be a positive integer (including 0)"
assert (args.initial_epoch >= 0 ), "Initial epoch needs to be at least 0"
assert (args.num_filters > 0), "Network needs more than one filter"
assert (args.num_res_blocks > 0), "Network needs more than one res block"
assert (args.reg_strength >= 0), "Regularization strength needs to be larger than or equal to 0"
assert (args.batch_size > 0), "Batch size needs to be larger than 0"
assert (args.initial_epoch < args.epochs), "Initial epoch is smaller than maximum number of epochs"
assert (args.period <= args.epochs), "Period to store the checkpoints is smaller " \
                                    "than maximum number of epochs"
assert ((args.scale % 2 == 0 or args.scale == 1) and args.scale <=8 and args.scale > 0), "Even number for scale"
assert (args.noiseLow <= args.noiseHigh), "Lower bound of the noise is higher than the higher bound"
if args.split_sr_from_cl:
    assert(args.split == 1 or args.split == 2), "Split must be 1 or 2 if split sr from cl"