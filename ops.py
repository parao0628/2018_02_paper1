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
    onehot = np.zeros((len(data),class_size))
    onehot[np.arange(len(data)), data] = 1.0
    return onehot

def getBoW(x_data, voca_size):
    x_bow = list()
    for data in x_data:
	tmp_bow = np.zeros(voca_size)
	for wid in data:
	    tmp_bow[wid] = data.count(wid)
	x_bow.append(tmp_bow)

    return x_bow

def getMinibatch(x_data, y_data, batch_idx, voca_size):
    mini_x = list()
    mini_y = list()

    for idx in batch_idx:
	mini_x.append(x_data[idx])
	mini_y.append(y_data[idx])
    
    #mini_bow = getBoW(mini_x, voca_size)
    #mini_bow = np.array(mini_bow)
    mini_x = np.array(mini_x)
    mini_y = np.array(mini_y)
    
    return mini_x, mini_y

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

