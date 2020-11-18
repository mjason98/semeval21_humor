import os
import torch 
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans

from .utils import MyBar, colorizar

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

def makeSiamData(data_path:str, ref_folder='data', humor_label='y_c'):
    '''
    This uses the files: 
        'pos_center.txt', 'pos_frontier.txt', 'neg_center.txt', 'neg_frontier.txt'
    to make the siames data, this files most be in ref_folder parameter.

    At least, the column humor_label most be in the file.
    '''
    files = ['pos_center.txt', 'pos_frontier.txt', 'neg_center.txt', 'neg_frontier.txt']
    vectors = []
    
    # Reading the centers and the frontiers
    for f in files:
        if not os.path.isfile(os.path.join(ref_folder, f)):
            print ('ERROR::FILE file', f, 'not found in', ref_folder)
            return
        else:
            vec = []
            with open(os.path.join(ref_folder, f), 'r') as file:
                for line in file.readlines():
                    vec.append(line.replace('\n', ''))
            vectors.append(vec)
    data = pd.read_csv(data_path)

    old_data,new_data, header = [], [], []
    for s in data.columns:
        old_data.append([])
        header.append(s)
    
    print ('# Making Siam data from', colorizar(os.path.basename(data_path)))
    bar = MyBar('data', max=len(data))
    for i in range(len(data)):
        p1, p2 = 1,2
        if int(data.loc[i, humor_label]) == 0:
            p1, p2 = 3, 0    
        for v in vectors[p1]:
            new_data.append(v)
            for h,p in zip(header, old_data):
                if h == humor_label:
                    p.append('1')
                else:
                    p.append(data.loc[i, h])
            for v in vectors[p2]:
                new_data.append(v)
                for h,p in zip(header, old_data):
                    if h == humor_label:
                        p.append('0')
                    else:
                        p.append(data.loc[i, h])
        bar.next()
    header.append('xr')
    bar.finish()
    del data 

    old_data.append(new_data)
    data = pd.concat([pd.Series(s) for s in old_data], axis=1)
    
    new_path = os.path.join('data', 'Siam'+os.path.basename(data_path))
    data.to_csv(new_path, index=None, header=header)
    return new_path

def makeSiam_ZData(data_path:str, model, ref_folder='data', batch=16):
    # files_pos = ['pos_center.txt', 'pos_frontier.txt']
    # files_neg = ['neg_center.txt', 'neg_frontier.txt']
    files_pos = ['pos_center.txt']
    files_neg = ['neg_center.txt']
    vectors_pos, vectors_neg = [], []
    new_name = os.path.join('data', 'Z'+os.path.basename(data_path))
    
    # Reading the centers and the frontiers
    for f in files_pos:
        if not os.path.isfile(os.path.join(ref_folder, f)):
            print ('ERROR::FILE file', f, 'not found in', ref_folder)
            return
        else:
            with open(os.path.join(ref_folder, f), 'r') as file:
                for line in file.readlines():
                    vectors_pos.append(line.replace('\n', ''))
    for f in files_neg:
        if not os.path.isfile(os.path.join(ref_folder, f)):
            print ('ERROR::FILE file', f, 'not found in', ref_folder)
            return
        else:
            with open(os.path.join(ref_folder, f), 'r') as file:
                for line in file.readlines():
                    vectors_neg.append(line.replace('\n', ''))
    pos_size, neg_size = len(vectors_pos), len(vectors_neg)
    vectors_pos = np.array([v for v in map(lambda x: [float(s) for s in x.split()], vectors_pos)], dtype=np.float32)
    vectors_neg = np.array([v for v in map(lambda x: [float(s) for s in x.split()], vectors_neg)], dtype=np.float32)
    vectors_np  = np.concatenate([vectors_pos, vectors_neg], axis=0)
    del vectors_neg
    del vectors_pos
    
    data = pd.read_csv(data_path)
    print ('# Making Z data from', colorizar(os.path.basename(data_path)))
    bar = MyBar('data', max=len(data)//batch)
    new_x = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch):
            end   = min(i+batch-1, len(data)-1)
            texts = data.loc[i:end, 'x'].to_numpy().tolist()
            texts = [np.array(v, dtype=np.float32) for v in map(lambda x: [float(s) for s in x.split()], texts)]
            texts = [ np.concatenate([v.reshape(1,-1)]*int(pos_size+neg_size), axis=0) for v in texts]

            texts = np.concatenate(texts, axis=0)
            vec   = np.concatenate([vectors_np]*int(end-i+1), axis=0)
            
            y_hat = model(torch.from_numpy(texts), torch.from_numpy(vec))
            y_hat = y_hat.reshape(-1,int(pos_size+neg_size)).numpy() #mirar esto
            new_x.append(y_hat)
            bar.next()
    bar.finish()
    new_x = np.concatenate(new_x, axis=0).tolist()
    new_x = [s for s in map(lambda x: ' '.join([str(v) for v in x]), new_x)]
    data.drop(['x'], axis=1, inplace=True)

    new_head = [s for s in data.columns] + ['x']
    data = pd.concat([data, pd.Series(new_x)], axis=1)
    data.to_csv(new_name, index=None, header=new_head)
    return new_name

def convert2EncoderVec(data_name:str, model, loader):
    model.train()
    new_name = os.path.join('data', data_name + '.csv')
    
    IDs, YC, YV, X = [], [], [], []

    print ('# Creating', colorizar(os.path.basename(new_name)))
    bar = MyBar('change', max=len(loader))
    with torch.no_grad():
        for data in loader:
            x = model(data['x'], ret_vec=True).numpy()
            try:
                y_c = data['y']
            except:
                y_c = None
            
            try:
                y_v = data['v']
            except:
                y_v = None
            
            try:
                ids = data['id']
            except:
                ids = None

            for i in range(x.shape[0]):
                l = x[i,:].tolist()
                X.append(' '.join([str(v) for v in l]))
                if y_c is not None:
                    YC.append(int(y_c[i].item()))
                if y_v is not None:
                    YV.append(y_v[i].item())
                if ids is not None:
                    IDs.append(ids[i].item())
            bar.next()
    bar.finish()

    conca, n_head = [], []
    if len(IDs) > 0:
        conca.append(pd.Series(IDs))
        n_head.append('id')
        del IDs
    if len(YC) > 0:
        conca.append(pd.Series(YC))
        n_head.append('y_c')
        del YC
    if len(YV) > 0:
        conca.append(pd.Series(YV))
        n_head.append('y_v')
        del YV
    conca.append(pd.Series(X))
    n_head.append('x')
    del X

    data = pd.concat(conca, axis=1)
    data.to_csv(new_name, index=None, header=n_head)
    return new_name