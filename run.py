from ops import *
from fasttext import *

import time
import random
import pickle
import argparse
import os

import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    
    #hyper params
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=10)
    parser.add_argument('--epoch', type=int, help='epoch size', default=5)
    parser.add_argument('--batch_size', type=int, help='batch size', default=100)
    parser.add_argument('--class_size', type=int, help='class size', default=4)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--scope', type=str, help='scope name', default='fasttext')
    parser.add_argument('--config', type=str, help='gpu or cpu', default='gpu')
    parser.add_argument('--seed', type=int, help='seed', default=0)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #tf verbose off(info, warning)
    
    if args.seed == 0:
        seed = int(time.time())%10000
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    #gpu setting
    gpu_config = tf.ConfigProto(device_count={'GPU':1})  # only use GPU no.1
    gpu_config.gpu_options.allow_growth = True # only use required resource(memory)
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 1 # restrict to 100%
    
    #cpu setting
    cpu_config = tf.ConfigProto()
    
    if args.config == 'gpu':
        config = gpu_config
    elif args.config == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        config = cpu_config
    
    x_data, y_data, word2ID = getData(args.class_size)

    model_class = Model(args, len(word2ID))
    model = model_class.classifier()
  
    with tf.Session(config=config) as sess: 
        sess.run(tf.global_variables_initializer())
        data_idx = {'train':range(len(x_data['train'])), 'test':range(len(x_data['test']))}
        for ep in range(args.epoch): 
            random.shuffle(data_idx['train'])
            start = time.time()
            for count in range(len(data_idx['train'])/args.batch_size+1):
                batch_idx = data_idx['train'][count*args.batch_size:(count+1)*args.batch_size]
                if not batch_idx:
                    continue
                x_minibatch, y_minibatch = getMinibatch(x_data['train'], y_data['train'], batch_idx, len(word2ID))
                
                feed_dict = {model.X: x_minibatch, 
                             model.Y: y_minibatch}
                tra, l = sess.run([model.train, model.loss], feed_dict=feed_dict)
                
                if count % 100 == 0:
                    print("[%d batch] | loss: %.2f | time: %.2f"%(count, l, time.time()-start))
            
            result = list()
            for count in range(len(data_idx['test'])/args.batch_size+1):
                batch_idx = data_idx['test'][count*args.batch_size:(count+1)*args.batch_size]
                if not batch_idx:
                    continue
                x_minibatch, y_minibatch = getMinibatch(x_data['test'], y_data['test'], batch_idx, len(word2ID))
                
                feed_dict = {model.X: x_minibatch, 
                             model.Y: y_minibatch}
                
                pred, loss = sess.run([model.pred, model.loss], feed_dict=feed_dict)
                result.extend(pred)
            
            score = accChk(result, y_data['test'])
            print("[%d epoch] | loss: %.2f | time: %.2f"%(ep, loss, time.time()-start))
            print("Score: %.2f"%score)
             
