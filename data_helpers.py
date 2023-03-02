#-*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix


class TrainDataset(object):
    def __init__(self, data_file,model,debug=False):
        self.t_user = []
        self.t_prd = []
        self.t_label = []
        self.model=model

        def read_data(data_file):
            with open(data_file, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    self.t_user.append(line[0])
                    self.t_prd.append(line[1])
                    self.t_label.append(float(line[2]))
                    if debug == True and len(self.t_user)>10:
                        break
        path_train =data_file
        read_data(path_train)
        self.data_size = len(self.t_user)
         
    def batch_iter(self, batch_size, num_epochs, shuffle=True):
        data_size = len(self.t_user)
        num_batches_per_epoch = int(data_size / batch_size) + (1 if data_size % batch_size else 0)
        self.t_user = np.asarray(self.t_user)
        self.t_prd = np.asarray(self.t_prd)
        self.t_label = np.asarray(self.t_label)



        for epoch in range(num_epochs):
            print('\n----------Epoch %04d -----------' % (epoch+1))
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                self.t_user = self.t_user[shuffle_indices]
                self.t_prd = self.t_prd[shuffle_indices]
                self.t_label = self.t_label[shuffle_indices]

            for batch_num in range(num_batches_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                user = list(map(int, self.t_user[start:end]))
                prd = list(map(int, self.t_prd[start:end]))
                label = list(map(float, self.t_label[start:end]))

                neg_user_item=[]
                neg_item_user=[]

                for u in user:
                    s = np.random.randint(len(self.neg_prdids[u]))
                    temp=self.neg_prdids[u][s]
                    neg_user_item.append(temp)

                for i in prd:
                    s = np.random.randint(len(self.neg_userids[i]))
                    temp=self.neg_userids[i][s]
                    neg_item_user.append(temp)

                result=[]
                result.append(np.array(user))
                result.append(np.array(prd))
                result.append(label)
                result.append(neg_user_item)
                result.append(neg_item_user)

                batch_data = result#ip(user, prd, label,neg_user_item,neg_item_user)
                yield batch_data



    def get_negative_sample(self,user_num, prd_num, test_user_item_matrix, test_item_user_matrix, neg_numbers):
        data_size=len(self.t_user)
        user = list(map(int, self.t_user))
        prd = list(map(int, self.t_prd))
        n_users = user_num
        n_items = prd_num
        train_matrix = csr_matrix((self.t_label, (user, prd)), shape=(n_users, n_items))
        all_items = set(np.arange(n_items))
        all_users = set(np.arange(n_users))
        neg_user_item_matrix = {}
        neg_item_user_matrix={}
        for u in range(n_users):
            test_items= set(test_user_item_matrix[u])
            negs= list(all_items - test_items - set(train_matrix.getrow(u).nonzero()[1]))[:neg_numbers]
            shuffle_indices = np.random.permutation(np.arange(len(negs)))
            neg_user_item_matrix[u] =[negs[x] for x in shuffle_indices]
        for i in range(n_items):
            test_users= set(test_item_user_matrix[i])
            negs = list(all_users - test_users- set(train_matrix.getcol(i).nonzero()[0]))[:neg_numbers]
            shuffle_indices = np.random.permutation(np.arange(len(negs)))
            neg_item_user_matrix[i]=[negs[x] for x in shuffle_indices]
        self.neg_prdids = neg_user_item_matrix
        self.neg_userids= neg_item_user_matrix

class TestDataset(object):
    """docstring for TestDataSet"""
    def __init__(self,data_file, debug=False):
        
        self.t_user = []
        self.t_prd = []
        self.t_label = []

        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                self.t_user.append(line[0])
                self.t_prd.append(line[1])
                self.t_label.append(float(line[2]))
                if debug == True and len(self.t_user)>1000:
                    break
        self.data_size = len(self.t_user)


    def predict_data(self, uids, pids):
        user = list(map(int, self.t_user))
        prd = list(map(int, self.t_prd))
        n_users = len(uids)
        n_items = len(pids)
        test_matrix = coo_matrix((self.t_label, (user, prd)), shape=(n_users, n_items))
        test_matrix = dok_matrix(test_matrix, (n_users, n_items))
        test_user_item_matrix = {}
        test_item_user_matrix ={}
        for u in range(n_users):
            test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]
        for i in range(n_items):
            test_item_user_matrix[i] = test_matrix.getcol(i).nonzero()[0]

        self.test_user_item_matrix= test_user_item_matrix
        self.test_item_user_matrix = test_item_user_matrix
        self.test_users = set([u for u in self.test_user_item_matrix.keys() if len(self.test_user_item_matrix[u]) > 0])
 



