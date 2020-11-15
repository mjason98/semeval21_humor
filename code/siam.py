import os
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans


def findCenter_and_Limits(data_path:str, K:int, M:int, method='k-mean'):
    '''
      For a cvs with vectors, find the K representatives an the M frotiers.
      This uses the method parameter to determine the algorithm to use in
      this process.

      data_path: file path to a csv with the 'x' column as vectors
                 the header in this csv most be equal to ('y_c', 'y_v', 'x')
      
      K: number of centers
      
      M: number of frontier's vectors
      
      method: the algorithm to use: ['k-means', 'graph']
    '''
    data = pd.read_csv(data_path)
    data.drop(['y_v'], axis=1, inplace=True)
    pos  = data.query('y_c == 1').drop(['y_c'], axis=1)
    neg  = data.query('y_c == 0').drop(['y_c'], axis=1)
    del data

    # finding the centers
    print ('# Calculating the centers')
    pos = pos.to_numpy().tolist()
    pos = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], pos)], dtype=np.float32)
    kmeans_pos = KMeans(n_clusters=K, random_state=0).fit(pos)    

    neg = neg.to_numpy().tolist()
    neg = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], neg)], dtype=np.float32)
    kmeans_neg = KMeans(n_clusters=K, random_state=0).fit(neg)    

    print ('# Calculating the frontier\'s vectors')
    pos_f, neg_f = [], []
    vec_size, mv = pos.shape[1], []
    for i in range(pos.shape[0]):
        vec = pos[i,:].reshape(-1,vec_size)
        vec = vec - kmeans_pos.cluster_centers_
        vec = np.sqrt((vec*vec).sum(axis=-1)).max()
        mv.append((vec, i))
    mv.sort()
    for _, i in mv[-M:]:
        pos_f.append(pos[i,:].tolist())
    del pos 
    mv.clear()
    
    for i in range(neg.shape[0]):
        vec = neg[i,:].reshape(-1,vec_size)
        vec = vec - kmeans_neg.cluster_centers_
        vec = np.sqrt((vec*vec).sum(axis=-1)).max()
        mv.append((vec, i))
    mv.sort()
    for _, i in mv[-M:]:
        neg_f.append(neg[i,:].tolist())
    del neg 
    del mv 

    kmeans_pos = kmeans_pos.cluster_centers_.tolist()
    kmeans_neg = kmeans_neg.cluster_centers_.tolist()
    
    with open(os.path.join('data', 'pos_center.txt'), 'w') as file:
        for l in kmeans_pos:
            file.write(' '.join([str(v) for v in l]) + '\n')
    with open(os.path.join('data', 'neg_center.txt'), 'w') as file:
        for l in kmeans_neg:
            file.write(' '.join([str(v) for v in l]) + '\n')
    with open(os.path.join('data', 'pos_frontier.txt'), 'w') as file:
        for l in pos_f:
            file.write(' '.join([str(v) for v in l]) + '\n')
    with open(os.path.join('data', 'neg_frontier.txt'), 'w') as file:
        for l in neg_f:
            file.write(' '.join([str(v) for v in l]) + '\n')

def makeSiamData(data_path:str, ref_folder='data'):
    '''
    This uses the files: 
        'pos_center.txt', 'pos_frontier.txt', 'neg_center.txt', 'neg_frontier.txt'
    to make the siames data, this files most be in ref_folder parameter.
    '''
    data_pos = pd.read_csv(data_path).query('y_c == 1')
    data_neg = pd.read_csv(data_path).query('y_c == 0')

    return

    # Making the 

    