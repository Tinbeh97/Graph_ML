# Graph-based Machine Learning for EEG data

The models presented in this represetory are mainly applied for biometric application. 

## Previous work

This represetory contains the codes for previous work [BrainPrint: EEG biometric identification based on analyzing brain connectivity graphs](https://www.sciencedirect.com/science/article/abs/pii/S0031320320301849). The code file is 'brainprint.py' and it uses graph features implemented in 'graphfeatures.py'. This model converts EEG signals into graph and manualy derive the features employing graph features such as minimum distance and clustering coefficients.

## Current work

Two novel machine learning method for automaticily deriving EEG signals' features is presented. [GCNN](GNN2.py) contains code for graph convolutional neural network for deriving brain graph features in supervised setting. [GVAE](VAE2.py) is a corresponding code for a novel graph-based variational auto-encoder. The GVAE can dervie an unsupervised brain graph embedding. 

## Prerequisites

All codes are written for Python 3 (https://www.python.org/) on Linux platform. The tensorflow version is 2.3.1.

The packages that are needed: tensorflow, os, sklearn, numpy, time, and networkx.

### Clone this repository

```
git clone git@github.com:Tinbeh97/Graph_ML.git
```
