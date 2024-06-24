
# coding: utf-8

# In[ ]:


import numpy as np
import deep_laa_support as dls
import random
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from sklearn.cluster import KMeans
from moco import moco_training_step
from moco import MoCoQueue, update_model_via_ema
import tensorflow.keras.backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EMBEDDING_DIM = 40
# queue = MoCoQueue(EMBEDDING_DIM, 256)       #256指队列长度，从embedding中取两个作为key   队列更新

# model_lb = tf.keras.Sequential(layers=[tf.keras.layers.Dense(256, activation='relu'),tf.keras.layers.Dense(40, activation=None)])
# model_ema_lb = tf.keras.Sequential(layers=[tf.keras.layers.Dense(256, activation='relu'),tf.keras.layers.Dense(40, activation=None)])
# model_ema_lb = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=(287, )),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(40, activation=None),
#     ]
# )
#
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=(287, )),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(40, activation=None),
#     ]
# )
#
# model_ema = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=(287, )),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(40, activation=None),
#     ]
# )
# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#        # self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')    #卷积,应该可以不用
#         #self.flatten = tf.keras.layers.Flatten()                         #展开,应该也不需要
#         # self.dense1 = self.add_weight(name='weight_vector1', shape=(287, 256),
#         #                               initializer='glorot_uniform', trainable=True,
#         #                               regularizer=None)
#         # self.dense2 = self.add_weight(name='weight_vector2', shape=(256, 40),
#         #                               initializer='glorot_uniform', trainable=True,
#         #                               regularizer=None)
#         self.d1 = tf.keras.layers.Dense(256, activation='relu')
#         self.d2 = tf.keras.layers.Dense(EMBEDDING_DIM, activation=None) #只需要后面的两层
#         #self.d3 = dls.moco_full_connect_relu_BN(moco_x, [feature_size, moco_h1_size_encoder])
#
#     def call(self, x):
#      #   x = self.conv1(x)                                       #删除
#        # x = self.flatten(x)                                     #删除
#         x = self.d1(x)
#         self.y = self.d2(x)
#         return self.y
#
#
# # Create an instance of the model
# model = MyModel()
# model_ema = MyModel()
# print("*************************")
# print(model_ema.layers[0])
# print("lb************")


# read data
# filename = 'default_file'
filename = 'feature_48_L'
data_all = np.load(filename +'.npz')
print('File ' + filename + '.npz ' 'loaded.')
user_labels = data_all['user_labels']         #标签集
true_labels = data_all['true_labels']
category_size = data_all['category_num']
print(category_size)
source_num = data_all['source_num']
print(source_num)
feature = data_all['feature']               #特征集
_, feature_size = np.shape(feature)
n_samples, _ = np.shape(true_labels)

def left_NN(input, training=None):
    flatten_input = tf.layers.flatten(input)
    print(flatten_input)
    flatten_input = tf.keras.layers.BatchNormalization(center=False, scale=False)(flatten_input)
    print(flatten_input)
    flatten_input =tf.layers.dense(inputs= flatten_input,units=128, activation='relu')
    x = tf.layers.dropout(flatten_input, training=None)
    print(x)
    x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
    cls_out = tf.nn.softmax(tf.layers.dense(inputs=x,units=2), axis=-1)
    return cls_out
# with tf.GradientTape() as tape:
#     feature_lb = left_NN(feature)
x_train = feature
# print(feature_lb)

print("***************")
# print(x_train.dtype)
x_train = x_train.astype(np.float32)
#x_train = x_train.batch(6033)


answers_bin_missings = []
for i in range(len(user_labels)):       #6033
    row = []
    for r in range(source_num):         #498
        row1 = []
        k1=2*r
        k2 = 2*r + 1
        row1.append(user_labels[i][k1])
        row1.append(user_labels[i][k2])
        row.append(row1)
    answers_bin_missings.append(row)
answers_bin_missings = np.array(answers_bin_missings, dtype=np.float32)

# Initialise the models and make the EMA model 90% similar to the main model  滑动模型 ,moco是以滑动窗口来进行的
# model(x_train[:1])              #取一个
# model_ema(x_train[:1])
# update_model_via_ema(model, model_ema, 0.1)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#
# optimizer = tf.keras.optimizers.Adam()

#================= basic parameters =====================
# define batch size (use all samples in one batch)
batch_size = n_samples # n_samples 例子个数 相当于一个batch
moco_batch_size = 128
cluster_num = 160
T = 1 # mc_sampling_times  采样次数
p_pure = np.array([0.5, 0.5], dtype=np.float32)

if np.max(feature) <= 1 and np.min(feature) >= 0:
    flag_node_type = 'Bernoulli'
else:
    flag_node_type = 'Gaussian'   
print(flag_node_type + ' output nodes are used.')

# 添加Moco部分#
# with tf.name_scope('moco'):
#     moco_x = tf.placeholder(dtype=tf.float32, shape=[moco_batch_size, feature_size], name='x_moco_input')  # (128,287)  随机抽取128个作为一个batch,同所有的进行比较  这个x的来源可能需要一个小的VAE网络
#
#     moco_h1_size_encoder = int(np.floor(feature_size / 2.0))  # 287/2
#     encoder_q = dls.moco_full_connect_relu_BN(moco_x, [feature_size, moco_h1_size_encoder])
#     encoder_q = dls.moco_full_connect(encoder_q, [moco_h1_size_encoder, 40])
#     encoder_q = tf.math.l2_normalize(encoder_q, axis = -1)
#     #momentum encoder
#     encoder_k = dls.moco_full_connect_relu_BN(moco_x, [feature_size, moco_h1_size_encoder])
#     encoder_k = dls.moco_full_connect(encoder_k, [moco_h1_size_encoder, 40])
#     encoder_k = tf.math.l2_normalize(encoder_k, axis = -1)
#
####添加对比学习部分,不使用Keras#########
#
# with tf.name_scope('moco_cla'):
#     moco_h1_size_encoder = int(np.floor(feature_size/2.0))  #287/2
#     moco_h2_size_encoder = 100
#     moco_embedding_size = 40
#     moco_h1_size_decoder = 100
#     moco_h2_size_decoder = int(np.floor(feature_size/2.0))
#     with tf.variable_scope('moco_encoder'):
#         x = tf.placeholder(dtype=tf.float32, shape=[batch_size, feature_size], name='x_input')  #(6033,287)
#         # mu_hx[batch_size, embedding_size]
#         # sigma_hx[batch_size, embedding_size]
#         with tf.variable_scope('feature_encoder_h1'):
#             _h1_encoder, w1_encoder, b1_encoder = dls.full_connect_relu_BN(x, [feature_size, moco_h1_size_encoder])  #tf.nn.relu(_tmp_results), weights, biases  此处使用了正态分布  w,x,b
#         with tf.variable_scope('feature_encoder_h2'):
#             _h2_encoder, w2_encoder, b2_encoder = dls.full_connect_relu_BN(_h1_encoder, [moco_h1_size_encoder, moco_h2_size_encoder])
#         with tf.variable_scope('feature_encoder_mu'):
#             mu_hx, w_mu_encoder, b_mu_encoder = dls.full_connect(_h2_encoder, [moco_h2_size_encoder, moco_embedding_size])            #正态分布中的𝜇 40
#         with tf.variable_scope('feature_encoder_sigma'):
#             sigma_hx, w_sigma_encoder, b_sigma_encoder = dls.full_connect(_h2_encoder, [moco_h2_size_encoder, moco_embedding_size])   #正态分布中的𝜎

def make_q(z, batch_size, alpha):
    sqd_dist_mat = pairwise_sqd_distance(z, batch_size)
    print(sqd_dist_mat.shape)
    q = tf.pow((1 + sqd_dist_mat / alpha), -(alpha + 1) / 2)
    print(q.shape)
    print(tf.zeros(shape=[batch_size]))
    q = tf.matrix_set_diag(q, tf.zeros(shape=[batch_size]))
    q = q / tf.reduce_sum(q, axis=0, keepdims=True)
    # q = 0.5*(q + tf.transpose(q))
    q = tf.clip_by_value(q, 1e-10, 1.0)

    return q

def pairwise_sqd_distance(X, batch_size):

    tiled = tf.tile(tf.expand_dims(X, axis=1), tf.stack([1, batch_size, 1]))
    tiled_trans = tf.transpose(tiled, perm=[1,0,2])
    diffs = tiled - tiled_trans
    sqd_dist_mat = tf.reduce_sum(tf.square(diffs), axis=2)
    sqd_dist_mat = tf.Print(sqd_dist_mat,[sqd_dist_mat])

    return sqd_dist_mat

def identity_init(shape):
    out = np.ones(shape, dtype=np.float32) * 0
    if len(shape) == 3:
        for r in range(shape[0]):
            for i in range(shape[1]):
                out[r, i, i] = 2
    elif len(shape) == 2:
        for i in range(shape[1]):
            out[i, i] = 2
    return out


def mig_loss_fuction(left_out, right_out):

    batch_num = left_out.shape[0]
    batch_num1 = tf.cast(batch_num, dtype=tf.float32)  #lb add

    I = tf.cast(np.eye(batch_num), dtype=tf.float32)
    E = tf.cast(np.ones((batch_num, batch_num)), dtype=tf.float32)
    normalize_1 = batch_num
    normalize_2 = batch_num * (batch_num - 1 )
    normalize_1 = tf.cast(normalize_1, dtype=tf.float32) #lb add
    normalize_2 = tf.cast(normalize_2, dtype=tf.float32) #lb add

    new_output = left_out / p_pure
    m = tf.matmul(new_output, right_out, transpose_b=True)
    noise = np.random.rand(1) * 0.0001
    m1 = tf.math.log(m * I + I * noise + E - I) # i<->i + i<->j. 此处E - I是为了让log为0，以便最大化P(i,i)互信息
    # m1 = tf.math.log(m * I + E - I)
    m2 = m * (E - I) # i<->j，最小化P(i,j)互信息
    # print(m)

    #loss 来自 KL，与MIG相反数
    return -(tf.reduce_sum(tf.reduce_sum(m1)) + batch_num1) / normalize_1 + tf.reduce_sum(tf.reduce_sum(m2)) / normalize_2
#================= encoder q(y|l) and q(h|x) =====================
with tf.name_scope('encoder'):
    #================= q(y|l) =====================标签集
    # define input l (source label vectors) 源标签向量
    input_size = source_num * category_size    #464*2 =user label大小
    with tf.variable_scope('q_yl'):
        l = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_size], name='l_input')   #6033,928
        pi_yl, weights_yl, biases_yl = dls.LAA_encoder(l, batch_size, source_num, category_size) #1,6033,2   权重,偏差w,b
    # loss: cross entropy between y_classifier and y_target for pre-training classifier
    with tf.variable_scope('q_yl'):
        pi_yl_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, category_size], name='pi_yl_target')  #6033,2
        loss_yl = dls.LAA_loss_classifier(pi_yl, pi_yl_target)     #交叉熵
    # optimizier
    learning_rate_yl = 0.01
    optimizer_pre_train_yl = tf.train.AdamOptimizer(learning_rate=learning_rate_yl).minimize(loss_yl)

    #================= q(h|x) =====================特征集  多元正态分布
    h1_size_encoder = int(np.floor(feature_size/2.0))  #287/2
    h2_size_encoder = 100
    embedding_size = 40
    h1_size_decoder = 100
    h2_size_decoder = int(np.floor(feature_size/2.0))

    with tf.variable_scope('q_hx'):
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, feature_size], name='x_input')  #(6033,287)
        # mu_hx[batch_size, embedding_size]
        # sigma_hx[batch_size, embedding_size]
        # with tf.variable_scope('feature_encoder_att'):
        #     _att_encoder, _, _ = dls.full_connect1(x, [feature_size, 128])  #tf.nn.relu(_tmp_results), weights, biases  此处使用了正态分布  w,x,b
        # with tf.variable_scope('feature_encoder_att'):
        #     _att1_encoder, _, _ = dls.full_connect1(_att_encoder, [256, 128])  #tf.nn.relu(_tmp_results), weights, biases  此处使用了正态分布  w,x,b
        with tf.variable_scope('feature_encoder_h1'):
            _h1_encoder, w1_encoder, b1_encoder = dls.full_connect_relu_BN(x, [feature_size, h1_size_encoder])  #tf.nn.relu(_tmp_results), weights, biases  此处使用了正态分布  w,x,b 287,287/2
        with tf.variable_scope('feature_encoder_h2'):
            _h2_encoder, w2_encoder, b2_encoder = dls.full_connect_relu_BN(_h1_encoder, [h1_size_encoder, h2_size_encoder])  #100
        with tf.variable_scope('feature_encoder_mu'):
            mu_hx, w_mu_encoder, b_mu_encoder = dls.full_connect(_h2_encoder, [h2_size_encoder, embedding_size])            #正态分布中的𝜇 40
        with tf.variable_scope('feature_encoder_sigma'):
            sigma_hx, w_sigma_encoder, b_sigma_encoder = dls.full_connect(_h2_encoder, [h2_size_encoder, embedding_size])   #正态分布中的𝜎
        # with tf.variable_scope('feature_softmax'):
        #     leftout = left_NN(x_train)
        #     print("lb leftout is",leftout)
            # mu_hx_sofamax, w_mu_encoder_softmax, b_mu_encoder_softmax = dls.full_connect(_h2_encoder, [h2_size_encoder, 64])    #lb:使用分类器进行分类,得到分类结果
        #     mu_hx_dropout=tf.nn.dropout(mu_hx_sofamax,0.5)
        # with tf.variable_scope('feature_softmax_result'):
        #     feature_softmax_result = dls.full_connect_softmax(mu_hx_dropout,[64, 2])
        # mu_hx, sigma_hx = dls.vae_encoder(x, feature_size, h1_size_encoder, h2_size_encoder, embedding_size)
        # embedding_h[batch_size, T, embedding_size]   这部分相当于是重参数化技术，将“采样”过程作为输入
        embedding_h = tf.reshape(mu_hx, [batch_size, 1, embedding_size])+ tf.reshape(sigma_hx, [batch_size, 1, -1])* tf.random_normal(shape=[batch_size, T, embedding_size], mean=0, stddev=1, dtype=tf.float32) #shape=(6033, 1, 40)
    # with tf.variable_scope('crowd_ann'):  # 添加人群推断right   #原有设计需要添加batchsize，这里暂时不添加 batchsize设定为1
    #     # y_ann = tf.placeholder(dtype=tf.float32, shape=[batch_size, feature_size], name='y_crowd')
    #     # leftout = left_NN(x_train)
    #     kernel = tf.Variable(identity_init((498, 2, 2)))
    #     crowd_answer = tf.transpose(answers_bin_missings, (1, 0, 2))   #batch_answers_bin_missings
    #     crowd_emb = tf.matmul(crowd_answer, kernel)
    #     agg_emb = tf.reduce_sum(crowd_emb, axis=0)
    #     type = 2
    #     out = 0
    #     if type == 1:
    #         print(agg_emb.shape)
    #         # print(feature_softmax_result.shape)
    #         print(p_pure.shape)
    #         out = agg_emb + tf.math.log(leftout + 0.001) + tf.math.log(p_pure)
    #     elif type == 2:
    #         out = agg_emb + tf.math.log(p_pure)
    #     elif type == 3:
    #         out = agg_emb + tf.math.log(leftout + 0.001)
    #     with tf.variable_scope('crowd_encoder_h1'):
    #         rightout = tf.nn.softmax(out,axis=-1)
    #     learning_rate = 1e-4
    #     # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #     loss_migmax = mig_loss_fuction(leftout, rightout)
    #     # vars = tf.GradientTape().watched_variables()
    #     # grads = tf.GradientTape().gradient(loss, vars)
    #     # grads = tf.gradients(loss, vars)   #lb add
    #     # optimizer.apply_gradients(zip(grads, vars))
    #     # tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_migmax)
    #     print("lb normal")
    with tf.variable_scope('q_hx_AE'):
        # x_reconstr, _, _ = dls.vae_decoder(mu_hx, embedding_size, h1_size_decoder, h2_size_decoder, feature_size)
        # with tf.variable_scope('feature_decoder_att1'):
        #     mu_hx, w1_decoder_att1, w1_decoder_att2 = dls.full_connect_relu_BN(mu_hx, [embedding_size, h1_size_decoder])
        # with tf.variable_scope('feature_decoder_att2'):
        #     mu_hx, _, _ = dls.full_connect1(mu_hx, [h1_size_decoder, embedding_size])
        with tf.variable_scope('feature_decoder_h1'):
            _h1_decoder, w1_decoder, b1_decoder = dls.full_connect_relu_BN(mu_hx, [embedding_size, h1_size_decoder])
        with tf.variable_scope('feature_decoder_h2'):
            _h2_decoder, w2_decoder, b2_decoder = dls.full_connect_relu_BN(_h1_decoder, [h1_size_decoder, h2_size_decoder])
        with tf.variable_scope('feature_decoder_rho'):
            if flag_node_type == 'Bernoulli':
                x_reconstr, w_rho_decoder, b_rho_decoder = dls.full_connect_sigmoid(_h2_decoder, [h2_size_decoder, feature_size])
            elif flag_node_type == 'Gaussian':
                x_reconstr, w_rho_decoder, b_rho_decoder = dls.full_connect(_h2_decoder, [h2_size_decoder, feature_size])
    # ####添加Z网络####层数应该为40-287-100-40
    # with tf.variable_scope('space_z'):
    #     with tf.variable_scope('space_z__h1'):
    #         z_h1_decoder, z_w1_decoder, z_b1_decoder = dls.full_connect_relu_BN(mu_hx, [embedding_size, feature_size])
    #     with tf.variable_scope('space_z__h2'):
    #         z_h2_decoder, z_w2_decoder, z_b2_decoder = dls.full_connect_relu_BN(z_h1_decoder,[feature_size, h2_size_decoder])
    #     with tf.variable_scope('space_z_rho'):
    #         if flag_node_type == 'Bernoulli':
    #             z_reconstr, z_w_rho_decoder, z_b_rho_decoder = dls.full_connect_sigmoid(z_h2_decoder,[h2_size_decoder, embedding_size])
    #         elif flag_node_type == 'Gaussian':
    #             z_reconstr, z_w_rho_decoder, z_b_rho_decoder = dls.full_connect(z_h2_decoder,[h2_size_decoder, embedding_size])
    #
    #     reconstr_loss = tf.reduce_mean(tf.squared_difference(x, x_reconstr), axis=1)
    #     batch_size1 = tf.shape(mu_hx)[0]
    #     p = make_q(mu_hx, batch_size1, alpha=20)
    #     q = make_q(z_reconstr, batch_size1, alpha=1)
    #     latent_loss = tf.reduce_sum(-(tf.multiply(p, tf.log(q))))
    #     # Joint loss.共同损失。
    #     joint_loss = tf.constant(1.0) * reconstr_loss + tf.constant(0.01) * latent_loss
        # Bernoulli 分类分布Cat
        loss_cross_entropy_AE = -tf.reduce_mean(tf.reduce_sum(x*tf.log(1e-10+x_reconstr) + (1.0-x)*tf.log(1e-10+(1.0-x_reconstr)), -1))       #交叉熵损失  每行求和，平均
        # loss_cross_entropy_AE1 = -tf.reduce_mean(tf.reduce_sum(mu_hx * tf.log(1e-10 + z_reconstr) + (1.0 - mu_hx) * tf.log(1e-10 + (1.0 - z_reconstr)),-1))  # 交叉熵损失  每行求和，平均
        # Gaussian 正态分布
        loss_square_AE = 0.5 * tf.reduce_mean(tf.square(x_reconstr - x))
        # loss_square_AE1 = 0.5 * tf.reduce_mean(tf.square(z_reconstr - mu_hx))
        constraint_w_AE = 0.5 * (tf.reduce_mean(tf.square(w1_encoder)) + tf.reduce_mean(tf.square(b1_encoder))
            + tf.reduce_mean(tf.square(w2_encoder)) + tf.reduce_mean(tf.square(b2_encoder))
            + tf.reduce_mean(tf.square(w_mu_encoder)) + tf.reduce_mean(tf.square(b_mu_encoder))
            + tf.reduce_mean(tf.square(w1_decoder)) + tf.reduce_mean(tf.square(b1_decoder))
            + tf.reduce_mean(tf.square(w2_decoder)) + tf.reduce_mean(tf.square(b2_decoder))
            + tf.reduce_mean(tf.square(w_rho_decoder)) + tf.reduce_mean(tf.square(b_rho_decoder)))            #KL
        if flag_node_type == 'Bernoulli':
            loss_AE = loss_cross_entropy_AE                 + constraint_w_AE                     #最终的 综合损失=重建损失loss_square_AE +KL损失constraint_w_AE
        elif flag_node_type == 'Gaussian':
            loss_AE = loss_square_AE                 + constraint_w_AE
        learning_rate_AE = 0.02
        optimizer_AE = tf.train.AdamOptimizer(learning_rate=learning_rate_AE).minimize(loss_AE)

    #================= p(x|h) =====================
    with tf.variable_scope('q_hx_AE'):
        with tf.variable_scope('feature_decoder_h1', reuse=True):
            _h_VAE = tf.reshape(embedding_h, [-1, embedding_size])     #embedding_h->q(h|x)
            _h1_decoder_VAE, _, _ = dls.full_connect_relu_BN(_h_VAE, [embedding_size, h1_size_decoder])
        with tf.variable_scope('feature_decoder_h2', reuse=True):
            _h2_decoder_VAE, _, _ = dls.full_connect_relu_BN(_h1_decoder_VAE, [h1_size_decoder, h2_size_decoder])
        with tf.variable_scope('feature_decoder_rho', reuse=True):
            if flag_node_type == 'Bernoulli':
                mu_xh, _, _ = dls.full_connect_sigmoid(_h2_decoder_VAE, [h2_size_decoder, feature_size])        #p(x|h),x中的每个元素都是伯努利分布 向量𝜌_𝑥(h) ∈ 〖["0,1" ]〗^𝐼通过权重为𝑤_(𝜌_𝑥 )的多层神经网络计算 就是𝜇
            elif flag_node_type == 'Gaussian':
                mu_xh, _, _ = dls.full_connect(_h2_decoder_VAE, [h2_size_decoder, feature_size])
            mu_xh = tf.reshape(mu_xh, [batch_size, T, feature_size])    #向量𝜌_𝑥(h)
            
    print('Encoders are constructed.')
    
#================= decoder p(l|y), p(x|h), p(y|z), p(h|z) and p(z) =====================
with tf.name_scope('decoder'):
    #================= p(l|y) =====================
    #每个工人n都有自己的分布参数 𝜋_𝑙^𝑛 (𝐲)，其中y作为输入，使用权重矩阵𝑤_(𝜋_𝑙)^𝑛的神经网络计算   计算每个工人的标签
    with tf.variable_scope('p_ly'):
        # pi_ly[category_size, 1, source_num*category_size]     y在encoder中进行计算q(y|l)
        pi_ly, weights_ly, biases_ly = dls.LAA_decoder(source_num, category_size)              #内嵌正态分布

        constraint_w_LAA = 0.5 * (tf.reduce_mean(tf.square(weights_ly)) + tf.reduce_mean(tf.square(biases_ly))
            + tf.reduce_mean(tf.square(weights_yl)) + tf.reduce_mean(tf.square(biases_yl)))                     #KL
        
    #================= p(y|z) =====================
    #在给定簇索引 z 的情况下绘制中间标签 y  p(y|z)=Cat(𝜋yz)
    with tf.variable_scope('p_yz'):
        # pi_yz[cluster_num, category_size]
        _pi_yz = tf.get_variable('pi_yz', dtype=tf.float32,
                                initializer=tf.random_normal(shape=[cluster_num, category_size], mean=0, stddev=1, dtype=tf.float32))  #p(y|z)=Cat(𝜋yz)  (200,2)  创建变量
        __pi_yz = tf.exp(_pi_yz)
        pi_yz = tf.div(__pi_yz, tf.reduce_sum(__pi_yz, -1, keepdims=True))
        
        pi_yz_assign = tf.placeholder(dtype=tf.float32, shape=[cluster_num, category_size], name='pi_yz_assign')
        initialize_pi_yz = tf.assign(_pi_yz, pi_yz_assign)
        
    #================= p(h|z) =====================
    with tf.variable_scope('p_hz'):
        # mu_hz[cluster_num, embedding_size]    p(h|z)集群z中绘制潜在的嵌入向量h 多元正态分布
        # sigma_hz[cluster_num, embedding_size]
        mu_hz = tf.get_variable('mu_hz', dtype=tf.float32, initializer=tf.random_normal(shape=[cluster_num, embedding_size], mean=0, stddev=1, dtype=tf.float32))   #定义𝜇 均值 （200,40）
        sigma_hz = tf.get_variable('sigma_hz', dtype=tf.float32, initializer=tf.ones([cluster_num, embedding_size], dtype=tf.float32))  #定义 𝜎 ,方差 (200,40)

        mu_hz_assign = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='mu_hz_assign')    #(200,40)  输入
        initialize_mu_hz = tf.assign(mu_hz, mu_hz_assign)                                                #赋值过程
        
    #================= p(z) =====================
    with tf.variable_scope('p_z'):
        # pi_z_prior[batch_size, cluster_num]
        pi_z_prior = tf.placeholder(dtype=tf.float32, shape=[batch_size, cluster_num], name='pi_z_prior')  #  p(z)=Cat(𝜋z)  (6033,200)   这是一个输入 通过计算可以得到一个初始值
        _pi_z = tf.get_variable('pi_z', dtype=tf.float32, initializer=tf.ones([batch_size, cluster_num]))
        __pi_z = tf.exp(_pi_z)
        pi_z = tf.div(__pi_z, tf.reduce_sum(__pi_z, -1, keepdims=True))

        pi_z_assign = tf.placeholder(dtype=tf.float32, shape=[batch_size, cluster_num], name='pi_z_assign')  #输入
        initialize_pi_z = tf.assign(_pi_z, pi_z_assign)
    print('Decoders are constructed.')
    
#================= elbo =====================
#ELBO整体可以看作是求log p(z,y,h,l,x)/q(z,y,h|l,x)  分子可以看作是decoder，分母为encoder
'''
q(h|x) log p(x|h)
q(y|l) log p(l|y)
q(h|x) log q(h|x)
q(y|l) log q(y|l)
q(z|x,l)q(h|x) log p(h|z)
q(z|x,l)q(y|l) log p(y|z)
q(z|x,l) log p(z)
q(z|x,l) log q(z|x,l)
q(z|x,l)
'''
with tf.name_scope('elbo'):
    #================= q(h|x) log p(x|h) =====================
    #q(h|x)  decoder中 计算了p(x|h),#向量𝜌_𝑥(h)-->mu_xh  encoder的q(h|x)和decoder的p(x|h)
    with tf.name_scope('q_hx_log_p_xh'):
        # reduce_mean along both T and batch_size
        _tmp = tf.reshape(x, [batch_size, 1, feature_size])
        if flag_node_type == 'Bernoulli':
            elbo_q_hx_log_p_xh = tf.reduce_mean(tf.reduce_sum(_tmp*tf.log(1e-10+mu_xh) + (1.0-_tmp)*tf.log(1e-10+(1.0-mu_xh)), -1))
        elif flag_node_type == 'Gaussian':
            elbo_q_hx_log_p_xh = -0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(_tmp-mu_xh), -1))
    
    #================= q(y|l) log p(l|y) =====================
    with tf.name_scope('q_yl_log_p_ly'):
        elbo_q_yl_log_p_ly = -dls.LAA_loss_reconstr(l, pi_ly, pi_yl)      #此处是encoder的q(y|l)和decoder的p(l|y)
        
    #================= q(h|x) log q(h|x) =====================     #此处是encoder的q(h|x)
    with tf.name_scope('q_hx_log_q_hx'):
        elbo_q_hx_log_q_hx = -0.5 * tf.reduce_mean(tf.reduce_sum(tf.log(1e-10+tf.square(sigma_hx)), -1))

    #================= q(y|l) log q(y|l) =====================    #encoder的q(y|l)
    with tf.name_scope('q_yl_log_q_yl'):
        elbo_q_yl_log_q_yl = tf.reduce_mean(tf.reduce_sum(pi_yl * tf.log(1e-10+pi_yl), -1))
    
    #================= q(z|x,l) =====================  这部分论文解释不清楚，暂时没看懂
    with tf.name_scope('q_zxl'):
        # p(h|z)[batch_size, T, cluster_num, 1]                #此部分通过从分布q(h|x)的T次采样ht来近似
        _h = tf.reshape(embedding_h, [batch_size, T, 1, embedding_size])   #(6033,1,1,40)
        _p_hz = -0.5 * tf.reduce_sum(
            tf.div(tf.square(_h-mu_hz), 1e-10+tf.square(sigma_hz)) 
            + tf.log(1e-10 + tf.square(sigma_hz)), -1, keepdims=True)  #p(h|z) (6033, 1, 200, 1)
        # p_zhy[batch_size, T, cluster_num, category_size]
        _p_zhy = tf.log(1e-10+pi_yz) + _p_hz + tf.log(1e-10+tf.reshape(pi_z, [batch_size, 1, cluster_num, 1]))       #此部分计算的是公式15中的p(z|y,h) ,其中p(y = k|z = c)， p(ℎ^𝑡  |z = c)， p(z = c)分别可以由对应的分布(3)，(5)，(2)直接计算
        #上面的三项分别为p(y|z) p(h|z) p(z)  (6033, 1, 200, 2)
        _p_zhy_max = tf.reduce_max(_p_zhy, 2, keepdims=True)
        #类似期望最大化方法中的E-step，通过最大值估计潜在变量分布
        p_zhy = tf.exp(_p_zhy - (_p_zhy_max + tf.log(1e-10+tf.reduce_sum(tf.exp(_p_zhy-_p_zhy_max), 2, keepdims=True)))) #(6033, 1, 200, 2)
        # q_zxl[batch_size, cluster_num]
        # reduce_mean along both category_size and T
        _q_zxl = tf.reduce_sum(tf.reshape(pi_yl, [batch_size, 1, 1, category_size]) * p_zhy, -1)  #求和  (6033, 1, 200)
        q_zxl = tf.reduce_mean(_q_zxl, 1)      #求平均 1/T   (6033, 200)
        
        # z_index[batch_size]
        z_index = tf.argmax(q_zxl, 1)     #(6033,)
        # cluster_pi_max[batch_size, category_size]
        # cluster_pi_avg[batch_size, category_size]
        cluster_pi_max = tf.gather(pi_yz, z_index)                  #(6033, 2)
        cluster_pi_avg = tf.matmul(q_zxl, pi_yz)                  #(6033, 2)
        
    #================= q(z|x,l)q(h|x) log p(h|z) =====================
    #================= q(h|x) log p(h|z) [batch_size, cluster_num] =====================
    with tf.name_scope('q_zxl_q_hx_log_p_hz'):
        # mu_hx[batch_size, embedding_size]
        # sigma_hx[batch_size, embedding_size]
        # mu_hz[cluster_num, embedding_size]
        # sigma_hz[cluster_num, embedding_size]
        _part_1 = tf.div(tf.square(tf.reshape(mu_hx, [batch_size, 1, embedding_size]) - mu_hz), 1e-10+tf.square(sigma_hz))    #q(h|x)
        _part_2 = tf.div(tf.square(tf.reshape(sigma_hx, [batch_size, 1, -1])), 1e-10+tf.square(sigma_hz))
        _part_3 = tf.log(1e-10 + tf.square(sigma_hz))
        # elbo_q_hx_log_p_hz[batch_size, cluster_num]
        elbo_q_hx_log_p_hz = -0.5 * tf.reduce_sum(_part_1 + _part_2 + _part_3, -1)
        elbo_q_zxl_q_hx_log_p_hz = tf.reduce_mean(tf.reduce_sum(q_zxl * elbo_q_hx_log_p_hz, -1))                                          #VAE函数损失第一项 最小化KL
    
    #================= q(z|x,l)q(y|l) log p(y|z) =====================
    #================= q(y|l) log p(y|z) [batch_size, cluster_num] =====================
    with tf.name_scope('q_zxl_q_yl_log_p_yz'):
        # pi_yz[cluster_num, category_size]
        # pi_yl[batch_size, category_size]
        # elbo_q_yl_log_p_yz[batch_size, cluster_num]
        elbo_q_yl_log_p_yz = tf.reduce_sum(tf.reshape(pi_yl, [batch_size, 1, category_size]) * tf.log(1e-10 + pi_yz), -1)
        elbo_q_zxl_q_yl_log_p_yz = tf.reduce_mean(tf.reduce_sum(q_zxl * elbo_q_yl_log_p_yz, -1))
    
    #================= q(z|x,l) log p(z) =====================
    #================= log p(z) [cluster_num] =====================
    with tf.name_scope('q_zxl_log_p_z'):
        # elbo_log_p_z[batch_size, cluster_num]
        elbo_log_p_z = tf.log(1e-10 + pi_z)
        elbo_q_zxl_log_p_z = tf.reduce_mean(tf.reduce_sum(q_zxl * elbo_log_p_z, -1))
    
    #================= q(z|x,l) log q(z|x,l) =====================
    with tf.name_scope('q_zxl_log_q_zxl'):
        # q_zxl[batch_size, cluster_num]
        elbo_q_zxl_log_q_zxl = tf.reduce_mean(tf.reduce_sum(q_zxl * tf.log(1e-10 + q_zxl), -1))
    
    #================= overall elbo ===================== KL
    elbo = elbo_q_hx_log_p_xh + elbo_q_yl_log_p_ly - elbo_q_hx_log_q_hx - elbo_q_yl_log_q_yl         + elbo_q_zxl_q_hx_log_p_hz + elbo_q_zxl_q_yl_log_p_yz + elbo_q_zxl_log_p_z - elbo_q_zxl_log_q_zxl    #KL公式展开
    
    q_zxl_entropy = -elbo_q_zxl_log_q_zxl
    
    with tf.variable_scope('regularization_prior'):                                                 #正则化     mu_hz_prior_mu  这一项需要变动换成对比学习的正则项
        mu_hz_prior_mu = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='mu_hz_prior_mu')
        # sigma_hz_prior_alpha = tf.placeholder(dtype=tf.float32, shape=[cluster_num, embedding_size], name='sigma_hz_prior')
        pi_yz_prior = tf.placeholder(dtype=tf.float32, shape=[cluster_num, category_size], name='pi_yz_prior')
        #约束
        constraint_prior = 0.5*tf.reduce_mean(tf.square(mu_hz - mu_hz_prior_mu))             - tf.reduce_mean(pi_yz_prior * tf.log(1e-10+pi_yz))             - tf.reduce_mean(pi_z_prior * tf.log(1e-10+pi_z))             + tf.reduce_mean(1.0*tf.log(1e-10+tf.square(sigma_hz))+tf.div(2.0, 1e-10+tf.square(sigma_hz)))

    loss_overall = -elbo         + constraint_w_AE         + constraint_w_LAA         + 1.0 * constraint_prior
    
    # optimizier
    learning_rate_overall = 0.001
    optimizer_overall = tf.train.AdamOptimizer(learning_rate=learning_rate_overall).minimize(loss_overall)

    print('Clustering-based label-aware autoencoder is constructed.')

saver = tf.train.Saver()


# In[ ]:


# for epoch in range(1000):
#     # K.clear_session()
#     tf.reset_default_graph()
#     # monitor_mu_hx = 0
#     # monitor_mu_hx1 = 0
#     # x_aug = 0
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         # _, monitor_loss_square_AE, monitor_mu_hx = sess.run([optimizer_AE, loss_square_AE, mu_hx],
#         #     feed_dict={x:feature})                                                #输入：特征  计算结果为q(h|x)中的h (6033, 40)   学习中间的嵌入层h-monitor_mu_hx
#         x_aug = x_train + 0.1 * tf.random.normal(tf.shape(x_train), dtype='float32')  # 使用添加噪音的方式替代VAE 作为图像数据集来说也不是不能这么用
#         print(x_train.shape)
#         print("lblb")
#         inital = model_ema_lb(x_train)
#         # print("lblblbblblbbis",inital)
#         # inital1 = model_ema_lb.predict(x_train)
#         # print("jjjjjjjjjj",inital1)
#         loss, monitor_mu_hx = moco_training_step(x_train, x_aug, queue, model, model_ema, optimizer)
#         #K.clear_session()
#         if epoch % 50 == 0:
#             print("epoch: {0} loss: {1}".format(epoch, loss))
#             #monitor_mu_hx1 = K.eval(monitor_mu_hx)
#             print("lb_over")
#             inital1 = model_ema.predict(x_train)
#             inital2 = model.predict(x_train)
#             print(inital1)
#             print(inital2)
    # result = model_ema_lb.predict(x_train)
    # print(x_train)
    # print("**********lb******")
    # print(result)
    # representation_model = model_ema(inputs=model_ema.inputs, outputs=model_ema.y)
    # dense_3_output = representation_model.predict(x_aug)
    #
    # print(dense_3_output.shape)
    # print(type(dense_3_output))
    # print(dense_3_output)
    # representation_layer = tf.keras.backend.function(inputs=[model_ema.layers[0].input], outputs=[model_ema.layers[1].output])
    # representation_layer = tf.keras.backend.function(inputs=[model_ema.call(x_aug)], outputs=[model_ema.layers[1].output])
    # representation = representation_layer([x_aug])
    # representation = np.array(representation)[0]
    #
    # print(representation.shape)
    # print(type(representation))
    # print(representation)

    #print(monitor_mu_hx1.shape)
# monitor_mu_hx1 = K.eval(monitor_mu_hx)

#================= training and inference =====================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #================= pre-train pi_yl =====================
    # assign batch variables (use whole data in one batch)
    # define majority voting regularizer
    majority_y = dls.get_majority_y(user_labels, source_num, category_size)             #计算估计的真值标签给出y


    # pre-train classifier
    #encoder
    print("Pre-train pi_yl ...")   #q(y|l)
    epochs = 50
    for epoch in range(epochs):
        _, monitor_pi_yl = sess.run([optimizer_pre_train_yl, pi_yl], 
            feed_dict={l:user_labels, pi_yl_target:majority_y})                    #输入：标签以及初始投票结果 计算结果为y的推断标签为q(y|l)的y
        if epoch % 10 == 0:
            hit_num = dls.cal_hit_num(true_labels, monitor_pi_yl)
            print("epoch: {0} accuracy: {1}".format(epoch, float(hit_num)/n_samples))  #查看准确率
    
    print("Pre-train hx_AE ...")
    epochs = 200
    for epoch in range(epochs):
        _, monitor_loss_square_AE, monitor_mu_hx = sess.run([optimizer_AE, loss_square_AE, mu_hx],
            feed_dict={x:feature})                                                #输入：特征  计算结果为q(h|x)中的h (6033, 40)   学习中间的嵌入层h-monitor_mu_hx   lb:加入左侧结果left result
        # x_aug = x_train + 0.1 * tf.random.normal(tf.shape(x_train), dtype='float32')  # 使用添加噪音的方式替代VAE 作为图像数据集来说也不是不能这么用
        # inital = model_ema_lb(x_train)
        # loss, monitor_mu_hx = moco_training_step(x_train, x_aug, queue, model, model_ema, optimizer)
        if epoch % 50 == 0:
            print("epoch: {0} loss: {1}".format(epoch, monitor_loss_square_AE))
            # K.clear_session()
            #print(monitor_mu_hx)
            # print(monitor_mu_hx[0])
            # print(majority_y[0])
            # print(np.shape(np.concatenate((monitor_mu_hx, majority_y), 1)))
            # print((np.concatenate((monitor_mu_hx, majority_y), 1))[0])
            # clustering_result = KMeans(n_clusters=cluster_num).fit(np.concatenate((monitor_mu_hx, majority_y), 1))
            # print(clustering_result.labels_)
            # print(np.shape(clustering_result.labels_))
    # monitor_mu_hx1 = K.eval(monitor_mu_hx)
    # monitor_mu_hx1 = model_ema.predict(x_train)
    # print(monitor_mu_hx1)
            #print(monitor_mu_hx1.shape)
            #================= calculate initial parameters计算初始参数 =====================此部分需要进行上面一些与训练的操作
        clustering_result = KMeans(n_clusters=cluster_num).fit(np.concatenate((monitor_mu_hx, majority_y), 1))    #拼接,聚类 cluster_num=200     聚类部分能否考虑优化？ (6033,42)
    print("kmeans over")
    labels_lb = clustering_result.labels_
    tsne = TSNE(n_components=2,random_state=0)
    X_tsne = tsne.fit_transform(np.concatenate((monitor_mu_hx, majority_y), 1))

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_lb, cmap='rainbow')
    plt.title("K-Means Clustering with t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar()
    plt.show()

    # 可视化前20个簇的降维结果
    # plt.figure(figsize=(8, 6))
    # for cluster_id in range(20):
    #     cluster_points = X_tsne[labels == cluster_id]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
    #
    # plt.title("K-Means Clustering with t-SNE Visualization")
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # plt.legend()
    # plt.show()

    # pi_z_prior_cluster = np.ones([n_samples, cluster_num]) / cluster_num
    pi_z_prior_cluster = dls.convert_to_one_hot(clustering_result.labels_, cluster_num, smooth=0.2)       #输入  (6033, 200) cluster_num=200    编码为onehot向量   clustering_result.labels_ :聚类后的分类情况200个簇里面 p(𝜋z)
    _ = sess.run(initialize_mu_hz, {mu_hz_assign:clustering_result.cluster_centers_[:, 0:embedding_size]})        #计算𝜇hz,p(𝜇hz)=N(𝜇2,I)
    # pi_yz_prior_cluster = np.ones([cluster_num, category_size]) / cluster_num
    pi_yz_prior_cluster = dls.get_cluster_majority_y(
        clustering_result.labels_, user_labels, cluster_num, source_num, category_size)       #通过收集集群中的用户标签，为每个集群返回y  (200, 2) 一共200个簇  簇标签
    _ = sess.run(initialize_pi_yz, {pi_yz_assign:pi_yz_prior_cluster})                          #初始化p(𝜋yz)=Dir(𝛽^𝑧)
    _ = sess.run(initialize_pi_z, {pi_z_assign:pi_z_prior_cluster})                             #初始化p(𝜋z)=Dir(𝛼^𝑚)

    mu_hz_prior_mu_cluster = clustering_result.cluster_centers_[:, 0:embedding_size]         #输入   簇中心

    predict_label = np.zeros([batch_size, category_size])     #(6033,2)   最终的初始化标签
    for i in range(batch_size):
        predict_label[i] = pi_yz_prior_cluster[clustering_result.labels_[i], :]
    print("Initial clustering accuracy: {0}".format(float(dls.cal_hit_num(true_labels, predict_label)) / n_samples))       #初始化时的准确率

    #================= save current model =====================
    saved_path = saver.save(sess, './my_model')


# In[ ]:
# K.clear_session()

with tf.Session() as sess:
    saver.restore(sess, './my_model')

    print("Train overall net ...")
    epochs = 5000
    for epoch in range(epochs):
        _, monitor_loss_overall, monitor_pi_yl, monitor_cluster_pi_max, monitor_cluster_pi_avg,             monitor_constraint_w_AE, monitor_constraint_prior = sess.run(
                [optimizer_overall, loss_overall, pi_yl, cluster_pi_max, cluster_pi_avg, constraint_w_AE, constraint_prior],
                feed_dict={l:user_labels, x:feature,
                           pi_z_prior:pi_z_prior_cluster,
                           mu_hz_prior_mu:mu_hz_prior_mu_cluster,
                           pi_yz_prior:pi_yz_prior_cluster})                  #h后三项计算loss使用
        if epoch % 10 == 0:
            print("epoch: {0} loss: {1}".format(epoch, monitor_loss_overall))
            print("epoch: {0} loss: {1}".format(epoch, monitor_constraint_w_AE))
            print("epoch: {0} loss: {1}".format(epoch, monitor_constraint_prior))
            hit_num_cluster_level_avg = dls.cal_hit_num(true_labels, monitor_cluster_pi_avg)
            print("epoch: {0} accuracy(cluster level avg): {1}".format(epoch, float(hit_num_cluster_level_avg)/n_samples))
    print("Training overall net. Done!")
