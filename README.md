# RSML-GCN
  Accurately identifying new indications for drugs is crucial in drug research and discovery. However, traditional drug indication discovery is costly and time-consuming. Computational drug repositioning provides an effective strategy for identifying potential drug–disease associations. This paper proposes a novel drug repositioning method named RSML-GCN, which employs reinforcement symmetric metric learning and graph convolutional network to predict potential drug indications. RSML-GCN first constructs a drug–disease heterogeneous network by integrating the association and feature information of drugs and diseases. Second, the graph convolutional network (GCN) is applied to complement the missing drug–disease association information. Third, reinforcement symmetric metric learning with adaptive margin is designed to learn the latent vector representation of drugs and diseases. Finally, new drug–disease associations based on the learned latent vector representation can be identified by the metric function. Comprehensive experiments on the benchmark dataset demonstrated the superior prediction performance of RSML-GCN to state-of-the-art methods for drug repositioning.  
  ![image](https://github.com/chlizhu/RSML-GCN/blob/main/images/fig.png)
# Description
  "data" contains the benchmark dataset used in RSML-GCN.  
  "data_helpers.py" is used to process training data and test data.  
  "SML_GCN.py" is the function of symmetric metric learning algorithm.  
  "gcn_model.py" is the function of GCN algorithm.
# Environment Requirement
  The code has been tested running under Python 3.7.11. The required packages are as follows:
  * numpy == 1.18.5
  * tensorflow == 1.15
  * scipy == 1.7.3
  * scikit-learn == 1.0.1
# Usage
  Please run train.py to reproduce the 10-fold cross-validation results reported in our paper. The users can also produce their cross-validation results by setting drug_sim, dis_sim and df in train.py to the user-provided data.
