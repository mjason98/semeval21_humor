#%%
import os, pandas as pd, math
import keras as K
import tensorflow as tf
import numpy as np
import sys
from utils import load_data, get_splits_for_val
from matplotlib import pyplot as plt
from sklearn import svm
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNet,Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)

method = 'ridge'
print(method)

DATA_PATH = '../data/train.csv'
text, is_humor, humor_rating, controversy, offensiveness = load_data(DATA_PATH)
text_dev, is_humor_dev, humor_rating_dev, controversy_dev, offensiveness_dev = load_data('../data/dev.csv')


bert_encodes = '../data/bertweet_'
roberta_encodes = '../data/roberta_'
xlnet_encodes = '../data/xlnet_'
features_path = '../data/features/encode_f_'

def load_encodes(phase):
    z = (np.load(bert_encodes + phase + '_encode.npy'), np.load(roberta_encodes + phase + '_encode.npy'),
         np.load(xlnet_encodes + phase + '_encode.npy'), np.load(features_path + phase + '.npy'))
    return z

def compute_mean_squared_error(y, yhat):
    loss = yhat - offensiveness_dev
    loss = np.sqrt(np.sum(loss**2)/len(offensiveness_dev))
    return loss

textb, textr, textx, features = load_encodes('train')
text = np.concatenate([ textb, textr, features ], axis=1)


textdevb, textdevr, textdevx, features_dev = load_encodes('dev')
textdev = np.concatenate([textdevb, textdevr, features_dev], axis=1)
iddev = pd.read_csv('../data/public_dev.csv').to_numpy()[:, 0]

texttestb, texttestr, texttestx, features_test = load_encodes('test')
texttest = np.concatenate([texttestb, texttestr, features_test], axis=1)
id_test = pd.read_csv('../data/public_test.csv').to_numpy()[:, 0]

# feat_sel = PCA(n_components=128)
# text = feat_sel.fit_transform(text)


print(text.shape)
splits = 5
data_val = get_splits_for_val(is_humor, splits)
allindex = [i for i in range(len(offensiveness))]

meanh = []
meano = []
meanc = []
meanhr = []

preds = {'humor':np.zeros((len(textdev), )), 'rating':np.zeros((len(textdev), )), 'cont':np.zeros((len(textdev), )),'off':np.zeros((len(textdev), )),}
preds_test = {'humor':np.zeros((len(texttest), )), 'rating':np.zeros((len(texttest), )), 'cont':np.zeros((len(texttest), )),'off':np.zeros((len(texttest), )),}

real = []
pred = []
for i in range(splits):
    train_index = list(set(allindex) - set(data_val[i]))
    test_index = list(data_val[i])

    # svchumor = svm.SVC(kernel ='linear')
    svroff = None
    # svcontr = svm.SVC(kernel ='linear')
    svrrating = None

    if method == 'randomforest':
        svroff = RandomForestRegressor()
        svrrating = RandomForestRegressor()
    elif method == 'svm':
        svroff = svm.SVR()
        svrrating = svm.SVR()
    elif method == 'ridge':
        svroff = RidgeCV()
        svrrating = RidgeCV()

    
    
    svroff.fit(text , list(offensiveness ))
    svrrating.fit(text , list(humor_rating ))
    
    off = svroff.predict(textdev)
    pred += list(off)
    real += list(offensiveness_dev)
    off = compute_mean_squared_error(offensiveness_dev , off)
    
    
    rating = svrrating.predict(textdev)
    rating = compute_mean_squared_error(humor_rating_dev, rating)
   
    meano.append(off)
    meanhr.append(rating)
    
    #--------------dev
    preds['off'] += np.array(svroff.predict(textdev))
    preds['rating'] += np.array(svrrating.predict(textdev))

    #--------------test
    preds_test['off'] += np.array(svroff.predict(texttest))
    preds_test['rating'] += np.array(svrrating.predict(texttest))

    print("humor_rating: {} offensiveness: {}".format(rating, off))
    break
splits = 1
print("-"*30 + 'measures')
print("humor_rating: {} offensiveness: {}".format( np.mean(meanhr), np.mean(meano)))

print("-"*30 + 'deviation')
print("humor_rating: {} offensiveness: {}".format( np.std(meanhr), np.std(meano)))

fig = plt.figure()
colors = ['b', 'g', 'r', 'y', 'w']
plt.scatter(real, pred, c = 'b', label = 'predictions')
plt.plot([i for i in range(6)], [i for i in range(6)], c = 'y', label = 'Pos')
plt.xlabel('real values')
plt.ylabel('predicted values')
plt.show()


# preds['humor'] = np.array(np.round(preds['humor']/splits), dtype=np.int)
preds['off'] = np.round(np.clip(preds['off']/splits, 0, 5), decimals=2)
preds['rating'] =  np.round(np.clip(preds['rating']/splits, 0, 5), decimals=2)
# preds['cont'] = np.array(np.round(preds['cont']/splits), dtype=np.int)

# preds_test['humor'] = np.array(np.round(preds_test['humor']/splits), dtype=np.int)
preds_test['off'] = np.round(np.clip(preds_test['off']/splits, 0, 5), decimals=2)
preds_test['rating'] =   np.round(np.clip(preds_test['rating']/splits, 0, 5), decimals=2)
# preds_test['cont'] = np.array(np.round(preds_test['cont']/splits), dtype=np.int)

#%%

# dictionary = {'id': iddev, 'humor_rating':preds['rating'],  'offense_rating':preds['off']}  
# df = pd.DataFrame(dictionary) 
# df.to_csv('../mlm.csv')
# print('Evaluation Done Dev!!') 

dictionary = {'id': id_test, 'humor_rating':preds_test['rating'], 'offense_rating':preds_test['off']}  
df = pd.DataFrame(dictionary) 
df.to_csv('../off_humor_rating.csv'.format(method))
print('Evaluation Done Test!!') 
# %%
