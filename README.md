# RSML-GCN
  Accurately identifying new indications for drugs is crucial in drug research and discovery. However, traditional drug indication discovery is costly and time-consuming. Computational drug repositioning provides an effective strategy for identifying potential drug–disease associations. This paper proposes a novel drug repositioning method named RSML-GCN, which employs reinforcement symmetric metric learning and graph convolutional network to predict potential drug indications. RSML-GCN first constructs a drug–disease heterogeneous network by integrating the association and feature information of drugs and diseases. Second, the graph convolutional network (GCN) is applied to complement the missing drug–disease association information. Third, reinforcement symmetric metric learning with adaptive margin is designed to learn the latent vector representation of drugs and diseases. Finally, new drug–disease associations based on the learned latent vector representation can be identified by the metric function. Comprehensive experiments on the benchmark dataset demonstrated the superior prediction performance of RSML-GCN to state-of-the-art methods for drug repositioning.
  ![image](https://github.com/chlizhu/RSML-GCN/blob/main/images/fig.png)
# Environment Requirement
  The code has been tested running under Python 3.7.11. The required packages are as follows:
  * numpy == 1.18.5
  * tensorflow == 1.15
  * scipy == 1.7.3
  * keras
# Usage
run train.py
