# Graph-based Machine Learning for EEG data
The models presented in this represetory are mainly applied for biometric application. 

##Previous work
This represetory contains the codes for previous work [BrainPrint: EEG biometric identification based on analyzing brain connectivity graphs](https://www.sciencedirect.com/science/article/abs/pii/S0031320320301849). The code file is 'brainprint.py' and it uses graph features implemented in 'graphfeatures.py'. This model converts EEG signals into graph and manualy derive the features employing graph features such as minimum distance and clustering coefficients.

##Current work
Two novel machine learning method for automaticily deriving EEG signals' features is presented. 'GNN2.py' contains code for graph convolutional neural network for deriving brain graph features in supervised setting. 'VAE2.py' is a corresponding code for a novel graph-based variational auto-encoder (GVAE). The GVAE can dervie an unsupervised brain graph embedding. 
