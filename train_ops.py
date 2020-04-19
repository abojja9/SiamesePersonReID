import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np


def conv2d(x, output_dim, ks=3, s=1, stddev=0.02, padding='SAME', name="conv2d", reuse=False):
    with tf.variable_scope(name):
        return tf.contrib.layers.conv2d(x, output_dim, ks, s, padding=padding, activation_fn=None, 
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        reuse=reuse)
def batch_norm(x, is_training, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, scope=name)
    
def relu(x, name="relu"):
    return tf.nn.relu(x, name=name)

def maxpool2d(x, pool_size=2, strides=2, padding='valid', name='maxpool2d'):
    return tf.layers.max_pooling2d(x, pool_size=pool_size, strides=strides, padding=padding, name=name)

def flatten(x, name="flatten"):
    return tf.contrib.layers.flatten(x, scope=name)

def dense(x, units=4096, activation=tf.sigmoid, name="dense"):
#     import pdb; pdb.set_trace()
    with tf.variable_scope(name):
        return tf.layers.dense(x, units, activation)


def model(input, is_training, batch_size, reuse=False):
    print(np.shape(input))
    with tf.name_scope("model"):
        # Group 1
        net = conv2d(input, 256, name='g_1_conv')
        print(np.shape(net))
        net = maxpool2d(net, name='g_1_maxpool')
        net = batch_norm(net, is_training, name='g_1_batchnorm')
        net = relu(net, name='g_1_relu')
        print(np.shape(net))
        
        # Group 2
        net = conv2d(net, 128, name='g_2_conv')
        print(np.shape(net))
        net = maxpool2d(net, name='g_2_maxpool')
        net = batch_norm(net, is_training, name='g_2_batchnorm')
        net = relu(net, name='g_2_relu')
        print(np.shape(net))
        
        # Group 3
        net = conv2d(net, 64, name='g_3_conv')
        print(np.shape(net))
        net = maxpool2d(net, name='g_3_maxpool')
        net = batch_norm(net, is_training, name='g_3_batchnorm')
        net = relu(net, name='g_3_relu')
        print(np.shape(net))
        
        # Group 4
        net = conv2d(net, 32, name='g_4_conv')
        print(np.shape(net))
        net = maxpool2d(net, name='g_4_maxpool')
        net = batch_norm(net, is_training, name='g_4_batchnorm')
        net = relu(net, name='g_4_relu')
        print(np.shape(net))
        
#         net = flatten(net)
        net = tf.reshape(net, [tf.shape(net)[0], 4096])
        print(np.shape(net))

        net = dense(net, 4096, activation=tf.sigmoid, name="model_dense_layer")
        print(np.shape(net))
        
    return net


def network(left_im, right_im, is_training, batch_size):
    with tf.variable_scope('feature_generator', reuse=tf.AUTO_REUSE) as sc:
        left_features = model(left_im, is_training=is_training, batch_size=batch_size)
        right_features = model(right_im, is_training=is_training, batch_size=batch_size)
        print ("[*] Model ran")
        merged_features = tf.abs(tf.subtract(left_features, right_features))
        logits = dense(merged_features, units=1, activation=None, name="last_dense")
        logits = tf.reshape(logits, [-1])
    return logits, left_features, right_features


def contrastive_loss(left_feat, right_feat, y, left_label, right_label, margin=0.2, use_loss=False):
    label = tf.equal(left_label, right_label)
    y = tf.cast(label, tf.float32)

    with tf.name_scope("contrastive_loss"):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(left_feat - right_feat, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance),
                                                       0))  # give penalty to dissimilar label if the distance is bigger than margin
        similarity_loss = tf.reduce_mean(dissimilarity + similarity) / 2
        if use_loss:
            tf.losses.add_loss(similarity_loss)
    return similarity_loss


def identity_loss(logits, left_label, right_label):
    label = tf.equal(left_label, right_label)
    label_float = tf.cast(label, tf.float32)

    logits = tf.cast(logits, tf.float32)
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_float)) #Using the mean distance
    tf.losses.add_loss(cross_entropy_loss)
    return cross_entropy_loss



def mean_average_precision(logits, left_label, right_label):
    label = tf.equal(left_label, right_label)
    label_float = tf.cast(label, tf.int64)

    logits = tf.cast(logits, tf.float32)
    average_precision = tf.compat.v1.metrics.average_precision_at_k(
        labels=label_float,
        predictions=logits,
        k=5
    )
    return average_precision
