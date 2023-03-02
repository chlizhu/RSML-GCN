
from utils import *
from gcn_model import GCNModel
from opt import Optimizer


def GCN(train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    adj = constructHNet(train_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_matrix.sum()
    X = constructNet(train_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model_2 = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model_2.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model_2,
            lr=lr, num_u=train_matrix.shape[0], num_v=train_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost, embeds = sess.run([opt.opt_op, opt.cost, model_2.embeddings], feed_dict=feed_dict)
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model_2.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


