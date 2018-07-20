from ops import *

import time
import random
import os

import tensorflow as tf

class Model:
    def __init__(self, args, voca_size):
	self.scope = args.scope
	self.hidden = args.hidden
	self.class_size = args.class_size
	self.feature_size = voca_size
	self.lr = args.lr

	self.X = tf.placeholder(tf.int32, [None, None], name='X')
	self.Y = tf.placeholder(tf.float32, [None, self.class_size], name='Y')
	self.embedding()

    def embedding(self):
	initializer = np.random.rand(self.feature_size, self.hidden).astype('float32')
	emb = tf.get_variable("emb", initializer=initializer[:], dtype=tf.float32)
	self.X_emb = tf.nn.embedding_lookup(emb, self.X)
	
    def classifier(self):
	with tf.variable_scope(self.scope) as scope:
	    self.inputs = tf.reduce_mean(self.X_emb, axis=1)
	    self.logit = tf.contrib.layers.fully_connected(inputs=self.inputs, num_outputs=self.class_size, activation_fn=None)
	    self.pred = tf.argmax(self.logit, axis=1, name='pred')
	    
	    self.loss = tf.losses.softmax_cross_entropy(self.Y, self.logit)
	    self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
	
	return self
