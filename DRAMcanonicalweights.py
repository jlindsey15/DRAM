#!/usr/bin/env python



import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
import random
from scipy import misc
import time
import sys

FLAGS = tf.flags.FLAGS

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


translated = True
if translated:
    dims = [100, 100]
else:
    dims = [28, 28]
img_size = dims[1]*dims[0]
read_n = 5
read_size = read_n*read_n
z_size=10
glimpses=10
batch_size=1
enc_size = 256
dec_size = 256
pretrain_iters=10000000
train_iters=10000000
learning_rate=1e-3
eps=1e-8
pretrain = False #False
classify = True #True
pretrain_restore = False
restore = True #True
rigid_pretrain = True
log_filename = "translatedplain/classify_weird_from_20000_log.csv"
load_file = "translatedplain/drawmodel20000.ckpt"
save_file = "translatedplain/classifymodel_weird_from_20000_"
draw_file = "translatedplain/zzzdraw_data_5000.npy"
start_non_restored_from_random = False


## BUILD MODEL ## 

REUSE=None

x = tf.placeholder(tf.float32,shape=(batch_size,img_size))
onehot_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))
lstm_enc = tf.nn.rnn_cell.LSTMCell(enc_size, read_size+dec_size) # encoder Op
lstm_dec = tf.nn.rnn_cell.LSTMCell(dec_size, z_size) # decoder Op

def normSquared(tensor, reduction_indices = None, name = None):
    squared_tensor = tf.square(tensor)
    euclidean_norm = (tf.reduce_sum(squared_tensor, tf.cast(reduction_indices, tf.int32)))
    return euclidean_norm_squared

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim])
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b



def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(dims[0]), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(dims[1]), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=REUSE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(dims[0]+1)/2*(gx_+1)
    gy=(dims[1]+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(dims[0],dims[1])-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma), params)


def read(x,h_dec_prev):
    Fx,Fy,gamma, params=attn_window("read",h_dec_prev,read_n)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,dims[1],dims[0]])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])
    x=filter_img(x,Fx,Fy,gamma,read_n) # batch x (read_n*read_n)
    return tf.concat(1,[x]), params # concat along feature axis



def write(h_dec):
    with tf.variable_scope("write",reuse=REUSE):
        return linear(h_dec,img_size)

def convertTranslated(images):
    newimages = []
    for k in xrange(batch_size):
        image = images[k, :]
        image = np.reshape(image, (28, 28))
        randX = random.randint(0, 72)
        randY = random.randint(0, 72)
        image = np.lib.pad(image, ((randX, 72 - randX), (randY, 72 - randY)), 'constant', constant_values = (0))
        image = np.reshape(image, (100*100))
        newimages.append(image)
    return newimages

def dense_to_one_hot(labels_dense, num_classes=10):
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



outputs=[0] * glimpses
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)
parameters_choices = []

for glimpse in range(glimpses):
    r, parameters=read(x,h_dec_prev)
    parameters_choices.append(parameters)
    with tf.variable_scope("encoder", reuse=REUSE):
        h_enc, enc_state = lstm_enc(tf.concat(1,[r,h_dec_prev]), enc_state)
    
    with tf.variable_scope("z",reuse=REUSE):
        z=linear(h_enc,z_size)

    with tf.variable_scope("decoder", reuse=REUSE):
        h_dec, dec_state = lstm_dec(z, dec_state)

    with tf.variable_scope("write", reuse=REUSE):
        outputs[glimpse] = linear(h_dec, img_size)
    h_dec_prev=h_dec
    REUSE=True

with tf.variable_scope("hidden1",reuse=None):
    hidden = tf.nn.relu(linear(h_dec_prev, 256))
with tf.variable_scope("hidden2",reuse=None):
    classification = tf.nn.softmax(linear(hidden, 10))
predquality = tf.log(classification + 1e-5) * onehot_labels
predquality = tf.reduce_mean(predquality, 0)
correct = tf.arg_max(onehot_labels, 1)
prediction = tf.arg_max(classification, 1)
R = tf.cast(tf.equal(correct, prediction), tf.float32)
reward = tf.reduce_mean(R)


def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))



def evaluate():
    data = mnist.input_data.read_data_sets("mnist", one_hot=True).test
    batches_in_epoch = len(data._images) // batch_size
    accuracy = 0
    
    for i in xrange(batches_in_epoch):
        nextX, nextY = data.next_batch(batch_size)
        if translated:
            nextX = convertTranslated(nextX)
        feed_dict = {x: nextX, onehot_labels:nextY}
        r = sess.run(reward, feed_dict=feed_dict)
        accuracy += r
    
    accuracy /= batches_in_epoch

    print("ACCURACY: " + str(accuracy))
    return accuracy


x_recons=tf.nn.sigmoid(outputs[-1])

reconstruction_loss=tf.reduce_sum(binary_crossentropy(x,x_recons),1)
reconstruction_loss=tf.reduce_mean(reconstruction_loss)


predcost = -predquality


##################


optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads=optimizer.compute_gradients(reconstruction_loss)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v)
train_op=optimizer.apply_gradients(grads)

varsToTrain = []

if True: #str2bool(sys.argv[1]):
    
    with tf.variable_scope("read",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)



if True: #str2bool(sys.argv[2]):
    
    with tf.variable_scope("encoder/LSTMCell",reuse=True):
        w = tf.get_variable("W_0")
        varsToTrain.append(w)
        b = tf.get_variable("B")
        varsToTrain.append(b)
            



if True: #str2bool(sys.argv[3]):
    
    with tf.variable_scope("z",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)



if True: #str2bool(sys.argv[4]):
    
    with tf.variable_scope("decoder/LSTMCell",reuse=True):
        w = tf.get_variable("W_0")
        varsToTrain.append(w)
        b = tf.get_variable("B")
        varsToTrain.append(b)



if True: #str2bool(sys.argv[5]):
    
    with tf.variable_scope("hidden1",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)


if True: #str2bool(sys.argv[6]):
    
    with tf.variable_scope("hidden2",reuse=True):
        w = tf.get_variable("w")
        varsToTrain.append(w)
        b = tf.get_variable("b")
        varsToTrain.append(b)












optimizer2=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
for v in varsToTrain:
    print(v.name)
grads2a=optimizer2.compute_gradients(predcost, var_list = varsToTrain)
grads2b=optimizer2.compute_gradients(predcost)

for i,(g,v) in enumerate(grads2a):
    if g is not None:
        grads2a[i]=(tf.clip_by_norm(g,5),v)
for i,(g,v) in enumerate(grads2b):
    if g is not None:
        grads2b[i]=(tf.clip_by_norm(g,5),v)
if rigid_pretrain:
    train_op2=optimizer2.apply_gradients(grads2a)
else:
    train_op2=optimizer2.apply_gradients(grads2b)


if not os.path.exists("mnist"):
    os.makedirs("mnist")
train_data = mnist.input_data.read_data_sets("mnist", one_hot=True).train


sess = tf.Session()
saver = tf.train.Saver()
with sess.as_default():
    tf.initialize_all_variables().run()
saver.restore(sess, "translatedplain/classifymodel_canonical_from_120000_0.ckpt")


sess2 = tf.Session()
saver2 = tf.train.Saver()
with sess2.as_default():
    tf.initialize_all_variables().run()
#saver.restore(sess2, "translatedplain/classifymodel_from_scratch_100000.ckpt")
    saver.restore(sess2, "translatedplain/classifymodel_from_scratch_120000_40000.ckpt")


xtrain, ytrain =train_data.next_batch(batch_size)
if translated:
    xtrain = convertTranslated(xtrain)
    
feed_dict={x:xtrain, onehot_labels:ytrain}
results=sess.run(h_dec,feed_dict)
results2 = sess2.run(h_dec, feed_dict)
hidden_fetched =np.reshape(results, [-1])
hidden_fetched2 = np.reshape(results2, [-1])

print(results)

print(np.dot(hidden_fetched, hidden_fetched2) / (np.linalg.norm(hidden_fetched) * np.linalg.norm(hidden_fetched2)))

print(np.linalg.norm(hidden_fetched - hidden_fetched2))


sum = 0
norm1Squared = 0
norm2Squared = 0
distSquared = 0

for v in varsToTrain:
    v1 = sess.run(tf.reshape(v, [-1]))
    v2 = sess2.run(tf.reshape(v, [-1]))
    print(v.name)
    #print(np.linalg.norm(v1-v2) / np.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    sum = sum + np.dot(v1, v2)
    norm1Squared = norm1Squared + np.square(np.linalg.norm(v1))
    norm2Squared = norm2Squared + np.square(np.linalg.norm(v2))
    distSquared = distSquared + np.square(np.linalg.norm(v1-v2))

print(sum / np.sqrt(norm1Squared * norm2Squared))
print(np.sqrt(distSquared))


'''
for v in varsToTrain:
    v1 = sess.run(tf.reshape(v, [-1]))
    v2 = sess2.run(tf.reshape(v, [-1]))
    #print(v.name)
    sum = sum + np.dot(v1, v2)
    distSquared = distSquared + np.square(np.linalg.norm(v1-v2))
    norm1Squared = norm1Squared + np.square(np.linalg.norm(v1))
    norm2Squared = norm2Squared + np.square(np.linalg.norm(v2))

print(sum / np.sqrt(norm1Squared * norm2Squared))
#print(np.sqrt(distSquared))


'''


sess.close()
sess2.close()
