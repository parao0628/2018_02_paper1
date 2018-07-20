from ops import *

import time
import random
import os

import tensorflow as tf

class Model:
    def __init__(self, args, voca_size):
        self.scope = args.scope
        self.hidden_size = args.hidden_size
        self.class_size = args.class_size
        self.feature_size = voca_size
        self.lr = args.lr

        self.X = tf.placeholder(tf.float32, [None, self.feature_size], name='X')
        self.Y = tf.placeholder(tf.float32, [None, self.class_size], name='Y')

    def embedding(self):
        initializer = np.random.rand(self.feature_size, self.hidden_size).astype('float32')
        emb = tf.get_variable("emb", initializer=initializer[:], dtype=tf.float32)
        self.X_emb = tf.nn.embedding_lookup(emb, self.X)
        
    def classifier(self):
        with tf.variable_scope(self.scope) as scope:
            
            self.hidden = tf.contrib.layers.fully_connected(inputs=self.X, 
                                                            num_outputs=self.hidden_size,
                                                            activation_fn=None,
                                                            scope='hvec')

            self.logit = tf.contrib.layers.fully_connected(inputs=self.hidden, 
                                                           num_outputs=self.class_size, 
                                                           activation_fn=None,
                                                           scope='logit')
            
            self.loss = tf.losses.softmax_cross_entropy(self.Y, self.logit)
            self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            
            self.pred = tf.argmax(self.logit, axis=1, name='pred')
        
        return self
