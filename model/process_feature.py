
import numpy as np
import os
import sys
import random
import gc

def getFeature(drug_dis_matrix, drug_matrix, dis_matrix):
    seed = 1
    np.random.seed(seed)
    epochs = 4000
    emb_dim = 64
    lr = 0.01
    adjdp = 0.6
    dp = 0.4

    print('getting gcn predict scores, it may take dozens of miniutes, please wait patiently......')

    from process_gcn import GCN

    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    for k in range(k_folds):
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = GCN(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
        del train_matrix
        gc.collect()

    return predict_y_proba

