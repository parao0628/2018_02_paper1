import pickle
import random
import numpy as np

def getData(class_size):
    PATH = './data/'
    x_data = pickle.load(open(PATH+'x_data.pickle','rb')) 
    y_data = pickle.load(open(PATH+'y_data.pickle','rb'))
    word2ID = pickle.load(open('./data/word2ID.pickle','rb'))
    
    y_onehot = dict()    
    for dtype in y_data:
	y_onehot[dtype] = getOneHot(y_data[dtype], class_size)

    return x_data, y_onehot, word2ID

def getOneHot(data, class_size):
    onehot = np.zeros((len(data),class_size), dtype='float32')
    onehot[np.arange(len(data)), data] = 1.0
    
    return onehot

def getBoW(x_data, voca_size):
    x_bow = list()
    for data in x_data:
	bow = np.zeros(voca_size, dtype='float32')
	x_len = len(data)
	for wid in data:
	    bow[wid] = float(data.count(wid))/float(x_len)
	
	x_bow.append(bow)

    return np.array(x_bow)

def getMinibatch(x_data, y_data, batch_idx, voca_size):
    mini_x = list()
    mini_y = list()

    for idx in batch_idx:
	mini_x.append(x_data[idx])
	mini_y.append(y_data[idx])
    
    mini_bow = getBoW(mini_x, voca_size)
    
    return mini_bow, mini_y

def accChk(result, y_data):
    correct = 0
    tot = len(result)
    assert tot == len(y_data)

    for idx, ans in enumerate(y_data):
	if result[idx] == ans:
	    correct += 1
	else:
	    continue
    
    acc = float(correct) / float(tot) * 100

    return acc

