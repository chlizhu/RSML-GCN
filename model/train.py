#-*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os, sys
from data_helpers import TrainDataset,TestDataset

import time
from utils import *
import math
from scipy.sparse import lil_matrix, dok_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
import SML_GCN
# import RankingMetrics


# Data loading params

# Model Hyperparameters
tf.flags.DEFINE_integer("hidden_size", 250, "hidden_size of rnn")
tf.flags.DEFINE_float("lr", 0.05, "Learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 800, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on test set after this many steps")
tf.flags.DEFINE_integer("verbose", 20, "Evaluate model on test set after this many steps")
tf.flags.DEFINE_boolean("debug",False, "debug")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("neg_numbers",200,"number of negative sampels")
tf.flags.DEFINE_string("dist","L2","L2 or L1")
tf.flags.DEFINE_string("model","SML","SML")
tf.flags.DEFINE_string("cuda","0","0 or 1")
tf.flags.DEFINE_float('gama',10,'gama')
tf.flags.DEFINE_float('dp',0.4,'dropout')
tf.flags.DEFINE_float('adjdp',0.6,'adj_dropout')
tf.flags.DEFINE_integer('simw', 6, "penalty factor of similarity")
FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# Load data
print("Loading data...")


drug_sim = np.loadtxt('data/drug_sim.csv', delimiter=',')
dis_sim = np.loadtxt('data/dis_sim.csv', delimiter=',')
df = np.loadtxt('data/drug_dis.csv', delimiter=',')
drug_dis_matrix = df
n_users, n_prds = drug_dis_matrix.shape
drug_dis_matrix = lil_matrix(drug_dis_matrix)
ass_matrix = dok_matrix(drug_dis_matrix.shape)
drug_sim_ = drug_sim.copy()
dis_sim_ = dis_sim.copy()
row, col = np.diag_indices_from(drug_sim_)
drug_sim_[row, col] = 1.0
row, col = np.diag_indices_from(dis_sim_)
dis_sim_[row, col] = 1.0
drug_matrix = drug_sim * FLAGS.simw
dis_matrix = dis_sim * FLAGS.simw
for u in range(n_users):
    prds = list(drug_dis_matrix.rows[u])
    for i in prds:
        ass_matrix[u, i] = 1
ass_matrix1 = sparse_to_tuple(ass_matrix)
drug_dis = ass_matrix1[0]
ass_matrix2 = list(ass_matrix1[0])

f = open('./data/drug_dis.txt', 'w')
for i in range(len(ass_matrix1[1])):
    f.write('{}\t{}\t{}\n'.format(ass_matrix2[i][0], ass_matrix2[i][1], ass_matrix1[1][i]))
f.close()


def get_dicts(uname, tr, te, val):
    ids = list(set(tr))
    print("The number of " + str(uname) + " in train dataset is " + str(len(ids)))
    ids1 = list(set(te))
    print("The number of ", uname, "in test dataset is ", len(ids1))
    ids2 = list(set(val))
    print("The number of ", uname, "in val dataset is ", len(ids2))
    out = ids + ids1 + ids2
    out = list(set(out))
    print("Total number of ", uname, "is ", len(out))
    return out


def get_model(uids, pids):
    if FLAGS.model == "SML":
        model = SML_GCN.model(
            hidden_size=FLAGS.hidden_size,
            user_num=len(uids),
            prd_num=len(pids),
            gama=FLAGS.gama,
            tra_matrix=train_drug_dis_matrix

        )
    return model


def get_feed_dict(model, data, type='train'):
    feed_dict = {}
    if type == 'train':
        u, p, y, neg_prdid, neg_userid= list(data[0]), list(data[1]), list(data[2]), list(data[3]), list(data[4])
        feed_dict = {
            model.userid: u,
            model.prdid: p,
            model.neg_prdid: neg_prdid,
            model.neg_userid: neg_userid,
        }
    elif type == 'test':
        u, p = zip(*data)
        feed_dict = {
            model.userid: u,
            model.prdid: p
        }
    return feed_dict

def predict(data):
    feed_dict = get_feed_dict(model, data, 'test')
    pred = -sess.run([model.pred_distance], feed_dict=feed_dict)[0]  # pred_distance 1*B
    return pred

def m_evaluation(t_test):
    pred_ratings = {}
    pred_ratings_5 = {}
    pred_ratings_10 = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []

    n_aupr_values = np.zeros([len(t_test.test_users), 1])
    n_auc_values = np.zeros([len(t_test.test_users), 1])
    num = - 1
    top = [5, 10]
    for u in t_test.test_users:
        num += 1
        userids = []
        user_neg_prds = trainset.neg_prdids[u]
        user_neg_prds = list(user_neg_prds) + list(t_test.test_user_item_matrix[u])

        prdids = []
        for j in user_neg_prds:
            prdids.append(j)
            userids.append(u)

        data = zip(userids, prdids)
        scores = predict(data)
        neg_item_index = list(zip(prdids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:top[0]]
        pred_ratings_10[u] = pred_ratings[u][:top[1]]

        p_5, r_5 = precision_recall_ndcg_at_k(top[0], pred_ratings_5[u],
                                                             t_test.test_user_item_matrix[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)

        p_10, r_10 = precision_recall_ndcg_at_k(top[1], pred_ratings_10[u],
                                                                 t_test.test_user_item_matrix[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)


        # recall, precision, auc, aupr
        y_true = []
        for i in np.arange(n_prds):
            if i in t_test.test_user_item_matrix[u]:
                y_true.append(1)
            else:
                y_true.append(0)
        y_true = np.array(y_true)[user_neg_prds]
        if np.sum(y_true) == 0:
            aupr_value = 0
            auc_value = 0
        else:
            precision_r, recall_r, threshold_r = precision_recall_curve(y_true, scores)

            aupr_value = auc(recall_r, precision_r)
            auc_value = roc_auc_score(y_true, scores)

        n_aupr_values[num] = aupr_value
        n_auc_values[num] = auc_value

    precision_5 = np.mean(p_at_5)
    precision_10 = np.mean(p_at_10)
    recall_5 = np.mean(r_at_5)
    recall_10 = np.mean(r_at_10)
    r_aupr = np.mean(n_aupr_values)
    r_auc = np.mean(n_auc_values)
    result = [r_aupr, r_auc, recall_5, precision_5, recall_10, precision_10]
    str_result = ["aupr" + "", "auc" + "", "r@" + str(top[0]) + "", "p@" + str(top[0]) + "", "r@" + str(top[1]) + "", "p@" + str(top[1])]
    return str_result, result, r_auc, recall_5



if __name__=="__main__":
    circle_time = 1
    for ii in range(circle_time):
        print("=======this is %dth circle=======" % (ii + 1))
        # 10-fold cross validation
        k_folds = 10
        shuffle_indices = np.random.permutation(np.arange(len(ass_matrix1[1])))
        random_index = drug_dis[shuffle_indices]
        ass_nums = len(ass_matrix1[1])
        CV_size = int(ass_nums / k_folds)
        temp = np.array(random_index[:ass_nums - ass_nums %
                                      k_folds]).reshape(k_folds, CV_size, -1).tolist()
        temp[k_folds - 1] = temp[k_folds - 1] + random_index[ass_nums - ass_nums % k_folds:].tolist()
        random_index = temp

        for k in range(k_folds):
            print("------this is %dth cross validation------" % (k + 1))
            test_index = random_index[k]
            f = open('./data/test.txt', 'w')
            for i in range(len(test_index)):
                f.write('{}\t{}\t{}\n'.format(test_index[i][0], test_index[i][1], ass_matrix1[1][i]))
            f.close()
            train_index = np.delete(random_index, k, axis=0)
            train_index = train_index.tolist()

            all_train = []
            for i in train_index:
                for j in i:
                    np.array(j)
                    all_train.append(j)

            all_train = np.array(all_train)

            if k != (k_folds - 1):
                shuffle_index = np.random.permutation(np.arange(ass_nums - CV_size))
                shuffle_train = all_train[shuffle_index]
                val_index = shuffle_train[:math.floor((ass_nums - CV_size) / 10)]
            else:
                shuffle_index = np.random.permutation(np.arange(ass_nums - CV_size - ass_nums % k_folds))
                shuffle_train = all_train[shuffle_index]
                val_index = shuffle_train[:math.floor((ass_nums - CV_size - ass_nums % k_folds) / 10)]

            all_train = all_train.tolist()
            val_index = val_index.tolist()
            train_index_ = [a for a in all_train if a not in val_index]
            all_train_data = np.array(train_index_)
            f = open('./data/dev.txt', 'w')
            for i in range(len(val_index)):
                f.write('{}\t{}\t{}\n'.format(val_index[i][0], val_index[i][1], ass_matrix1[1][i]))
            f.close()
            testset = TestDataset('./data/test.txt', FLAGS.debug)
            valset = TestDataset('./data/dev.txt', FLAGS.debug)

            train_drug_dis_matrix = sp.coo_matrix(
                (df[all_train_data[:, 0], all_train_data[:, 1]], (all_train_data[:, 0], all_train_data[:, 1])),
                shape=(drug_matrix.shape[0], dis_matrix.shape[0])).toarray()
            from process_feature import getFeature

            res_drug_dis = getFeature(train_drug_dis_matrix, drug_matrix, dis_matrix)
            index_ass = np.where(train_drug_dis_matrix == 1)
            res_drug_dis[index_ass] = 1
            index = np.where(res_drug_dis == 1)
            res_drug_dis = np.where(res_drug_dis < 0.5, 0, res_drug_dis)
            drug_dis_2 = dok_matrix(res_drug_dis)
            drug_dis_2 = sparse_to_tuple(drug_dis_2)
            res_all_train = drug_dis_2[0].tolist()
            f = open('./data/train.txt', 'w')
            for i in range(len(res_all_train)):
                f.write('{}\t{}\t{}\n'.format(res_all_train[i][0], res_all_train[i][1],
                                              res_drug_dis[res_all_train[i][0]][res_all_train[i][1]]))
            f.close()
            trainset = TrainDataset('./data/train.txt', FLAGS.model, FLAGS.debug)

            pids = get_dicts("product", trainset.t_prd, testset.t_prd, valset.t_prd)
            uids = get_dicts("user", trainset.t_user, testset.t_user, valset.t_user)
            print('pids', len(pids), 'uids', len(uids))
            user_num = len(uids)
            prd_num = len(pids)
            print(user_num, type(user_num))

            testset.predict_data(uids, pids)
            valset.predict_data(uids, pids)
            trainset.get_negative_sample(user_num, prd_num, testset.test_user_item_matrix,
                                         testset.test_item_user_matrix, FLAGS.neg_numbers)
            trainbatches = trainset.batch_iter(FLAGS.batch_size, FLAGS.num_epochs)

            print("Loading data finished...")

            """Train"""
            with tf.Graph().as_default():
                os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda
                session_config = tf.ConfigProto()
                # session_config.gpu_options.per_process_gpu_memory_fraction = 0.65
                session_config.gpu_options.allow_growth = True
                sess = tf.Session(config=session_config)
                with sess.as_default():
                    model = get_model(uids, pids)
                    model.build_network()
                    # Define Training procedure
                    global_step = tf.Variable(0, name="global_step", trainable=False)
                    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

                    optimizer = tf.train.AdagradOptimizer(FLAGS.lr).minimize(model.loss, global_step=global_step)
                    timestamp = str(int(time.time()))
                    sess.run(tf.global_variables_initializer())
                    # Training loop. For each batch...
                    all_train_loss = []
                    train_loss_value = []
                    pre_recall = 0
                    patience = 20
                    epoch_patience = 0
                    epoch_best = 0
                    stop_threshold = 0.0015

                    for tr_batch in trainbatches:
                        feed_dict = get_feed_dict(model, tr_batch, "train")
                        start_time = time.time()
                        if FLAGS.model == 'SML':
                            _, step, loss, _, _, b, b1 = sess.run(
                                [optimizer, global_step, model.loss, model.clip_U, model.clip_P, model.clip_B,
                                 model.clip_B1], feed_dict)
                            all_train_loss.append(loss)
                        if math.isnan(loss):
                            print("loss =NAN")
                            sys.exit()
                        if step % FLAGS.verbose == 0:
                            print("time={}, step {}, loss {:g}".format(time.time() - start_time, step, loss))

                        train_data_size = len(trainset.t_user)
                        one_epoch_num_batches = int(train_data_size / FLAGS.batch_size) + (
                            1 if train_data_size % FLAGS.batch_size else 0)
                        if step % one_epoch_num_batches == 0:
                            ave_train_loss = np.mean(all_train_loss)
                            all_train_loss = []
                            # Each epoch validation set for validation
                            print('============Validation============')
                            str_result, results1, val_auc, val_recall = m_evaluation(valset)
                            print(str_result)
                            print(results1)
                            if abs(val_recall - pre_recall) >= stop_threshold:
                                epoch_patience = 0
                                pre_recall = val_recall
                                epoch_best = step % one_epoch_num_batches

                                print('train_loss:  %s\n' % loss,
                                      'aupr_val:   %s\n' % results1[0],
                                      'auc_val:    %s\n' % results1[1],
                                      'pre_recall:    %s\n' % pre_recall)
                            else:
                                epoch_patience += 1
                                if epoch_patience > patience:
                                    print("\nStopp Training!")
                                    break
                    print('======================Test======================')
                    _, results2, _, _ = m_evaluation(testset)
                    print(str_result)
                    print(results2)
 

        
 

        






