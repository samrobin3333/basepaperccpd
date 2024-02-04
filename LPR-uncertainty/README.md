# Benchmarking Probabilistic Deep Learning Methods for License Plate Recognition

This repository contains the implementation of the paper

F. Schirrmacher, B. Lorch, A. Maier and C. Riess, "Benchmarking Probabilistic Deep Learning Methods for License Plate Recognition," in IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 9, pp. 9203-9216, Sept. 2023, doi: 10.1109/TITS.2023.3278533. [IEEE]([https://ieeexplore.ieee.org/abstract/document/9191253](https://ieeexplore.ieee.org/abstract/document/10143386)) 

If you use this code in your work, please cite:

      @ARTICLE{10143386,
        author={Schirrmacher, Franziska and Lorch, Benedikt and Maier, Anatol and Riess, Christian},
        journal={IEEE Transactions on Intelligent Transportation Systems}, 
        title={Benchmarking Probabilistic Deep Learning Methods for License Plate Recognition}, 
        year={2023},
        volume={24},
        number={9},
        pages={9203-9216},
        doi={10.1109/TITS.2023.3278533}}


## Getting started

To download the code, fork the repository or clone it using the following command:

```
  git clone https://github.com/franziska-schirrmacher/LPR-uncertainty.git
```

### Requirements

- python 3.7
- keras 2.3.1
- tensorflow 1.14
- h5py 2.10.0

### Code structure

- **checkpoints**: This folder contains the stored weights (needs to be created)

- **data**: This folder contains all datasets for each of the experiments (needs to be created)

- **src**: This folder contains the source code of the experiments and the proposed architectur



### Datasets

In order to reproduce the results, you need to download the [MNIST](http://yann.lecun.com/exdb/mnist/), the [SVHN](http://ufldl.stanford.edu/housenumbers/) and the [CCPD](https://github.com/detectRecog/CCPD) datset.
Please check out https://github.com/franziska-schirrmacher/SR2 for details regarding the MNIST and SVHN dataset. The code base is the same.
