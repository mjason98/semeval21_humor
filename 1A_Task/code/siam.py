import os
import re
import random
import torch 
import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support as score
import networkx as nx

from .utils import MyBar, colorizar, headerizar

# Path to infomap gitHub code--------------------------------------
# Before run this code, the infomap code most be compiled

INFOMAP_PATH = '/DATA/work_space/2-AI/3-SemEval21/infomap-master'
INFOMAP_EX   = 'Infomap'

# -----------------------------------------------------------------
def setInfomapData(p1, p2):
    global INFOMAP_EX
    global INFOMAP_PATH
    INFOMAP_PATH = p1
    INFOMAP_EX   = p2

def completeNodes(reference:list, store:list, vects:list):
    Len = len(reference)
    reference.sort()
    p = 0
    
    for i in range(len(vects)):	
        if p >= Len or i < reference[p]:
            store.append(vects[i])
        while p < Len and reference[p] == i:
            p += 1

def findCenter_and_Limits(data_path:str, K:int, M:int, method='k-means', method_distance='euclidea', eps=1e-7, umbral = 0.05, max_module=1):
    '''
      For a cvs with vectors, find the K representatives an the M frotiers.
      This uses the method parameter to determine the algorithm to use in
      this process.

      data_path: file path to a csv with the 'x' column as vectors
                 the header in this csv most be equal to ('y_c', 'y_v', 'x')
      
      K: number of centers
      
      M: number of frontier's vectors
      
      method: the algorithm to use: ['k-means', 'i-graph', 'c-graph']
      
      method_distance: Distance function to use in the method: ['euclidea', 'cosine']

      max_module: maximun nodes to use per module using infomap algorithm ('i-graph')

      umbral: float or 2-tuple (humor threshold, not humor threshold)
        If i-graph is selected, this represents a percent of the minor edges to use; if
        c-graph is selected, and an endge has a weight higer of this, this edge is descarted
    '''
    Me = ['k-means', 'i-graph', 'c-graph']
    Me_d = ['euclidea', 'cosine']
    if method not in Me:
        print('ERROR::parameter method not in', '['+', '.join(Me)+'].')
        return
    if method_distance not in Me_d:
        print('ERROR::parameter method_distance not in', '['+', '.join(Me_d)+'].')
        return

    data = pd.read_csv(data_path)
    data.drop(['humor_rating', 'id'], axis=1, inplace=True)
    pos  = data.query('is_humor == 1').drop(['is_humor'], axis=1)
    neg  = data.query('is_humor == 0').drop(['is_humor'], axis=1)
    del data

    pos_c, neg_c = [], []
    if method == 'k-means':
        if type(umbral) is not float:
            umbral = umbral[0]
        # finding the centers
        print ('# Calculating the centers')
        pos = pos.to_numpy().tolist()
        pos = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], pos)], dtype=np.float32)
        kmeans_pos = KMeans(n_clusters=K, random_state=0).fit(pos)    

        neg = neg.to_numpy().tolist()
        neg = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], neg)], dtype=np.float32)
        kmeans_neg = KMeans(n_clusters=K, random_state=0).fit(neg)    

        print ('# Calculating the frontier\'s vectors')
        vec_size, mv = pos.shape[1], []
        for i in range(pos.shape[0]):
            vec = pos[i,:].reshape(-1,vec_size)
            vec = vec - kmeans_pos.cluster_centers_
            vec = np.sqrt((vec*vec).sum(axis=-1)).mean()
            mv.append((vec, i))
        mv.sort()
        for _, i in mv[-M:]:
            pos_c.append(pos[i,:].tolist())
        for _, i in mv[:K]:
            pos_c.append(pos[i,:].tolist())
        del pos 
        mv.clear()
        
        for i in range(neg.shape[0]):
            vec = neg[i,:].reshape(-1,vec_size)
            vec = vec - kmeans_neg.cluster_centers_
            vec = np.sqrt((vec*vec).sum(axis=-1)).mean()
            mv.append((vec, i))
        mv.sort()
        for _, i in mv[-M:]:
            neg_c.append(neg[i,:].tolist())
        for _, i in mv[:K]:
            neg_c.append(neg[i,:].tolist())
        del neg 
        del mv
    elif method == 'i-graph':
        if type(umbral) is float:
            umbral = (umbral, umbral)
        if not os.path.isdir(INFOMAP_PATH):
            print ('ERROR::PATH the path', INFOMAP_PATH, 'does not exist!, This function will be skiped')
            return
        print ('Making graphs, the threshold will be calculated with a {:.3}%, {:.3}% of positive and negative edges'.format(umbral[0]*100, umbral[1]*100))
        bar = MyBar('i-graph', max=len(pos)+len(neg))
        
        pos_name = os.path.join('data', 'pos_graf_'+method_distance)
        pos = pos.to_numpy().tolist()
        pos = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], pos)], dtype=np.float32)
        pos_min, pos_max, pos_umb = None, None, None
        
        with open(pos_name+'t', 'w') as file:
            for i in range(pos.shape[0]):
                tmp_v = pos[i].reshape(1, -1)
                if method_distance == 'euclidea':
                    tmp_v = np.sqrt(((pos - tmp_v)**2).sum(axis=-1))
                elif method_distance == 'cosine':
                    n1 = np.sqrt((tmp_v**2).sum(axis=-1))
                    n2 = np.sqrt((pos**2).sum(axis=-1))
                    tmp_v = (tmp_v*pos).sum(axis=-1)
                    tmp_v = tmp_v/(n1*n2 + eps)
                tmp_v = tmp_v.reshape(-1)
                for j in range(tmp_v.shape[0]):
                    if j == i:
                        continue
                    file.write(' '.join([str(i+1), str(j+1), str(tmp_v[j])])+'\n')
                    if pos_min is None:
                        pos_min = tmp_v[j]
                    else:
                        pos_min = min(pos_min, tmp_v[j])
                    if pos_max is None:
                        pos_max = tmp_v[j]
                    else:
                        pos_max = max(pos_max, tmp_v[j])
                bar.next()
        pos_umb = (1 - umbral[0])*pos_min + umbral[0]*pos_max


        neg_name = os.path.join('data', 'neg_graf_'+method_distance)
        neg = neg.to_numpy().tolist()
        neg = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], neg)], dtype=np.float32)
        neg_min, neg_max, neg_umb = None, None, None
        
        with open(neg_name+'t', 'w') as file:
            for i in range(neg.shape[0]):
                tmp_v = neg[i].reshape(1, -1)
                if method_distance == 'euclidea':
                    tmp_v = np.sqrt(((neg - tmp_v)**2).sum(axis=-1))
                elif method_distance == 'cosine':
                    n1 = np.sqrt((tmp_v**2).sum(axis=-1))
                    n2 = np.sqrt((neg**2).sum(axis=-1))
                    tmp_v = (tmp_v*neg).sum(axis=-1)
                    tmp_v = tmp_v/(n1*n2 + eps)
                tmp_v = tmp_v.reshape(-1)
                for j in range(tmp_v.shape[0]):
                    if j == i:
                        continue
                    file.write(' '.join([str(i+1), str(j+1), str(tmp_v[j])])+'\n')
                    if neg_min is None:
                        neg_min = tmp_v[j]
                    else:
                        neg_min = min(neg_min, tmp_v[j])
                    if neg_max is None:
                        neg_max = tmp_v[j]
                    else:
                        neg_max = max(neg_max, tmp_v[j])
                bar.next()
        neg_umb = (1 - umbral[1])*neg_min + umbral[1]*neg_max
        del neg_max
        del neg_min
        bar.finish()

        print ('Filtering using thresholds positive: {:.3}, negative: {:.3}'.format(pos_umb, neg_umb))
        with open(neg_name+'t', 'r') as R, open(neg_name, 'w') as W:
            for line in R:
                v = float(line.replace('\n', '').split()[-1])
                if v < neg_umb:
                    W.write(line)
        with open(pos_name+'t', 'r') as R, open(pos_name, 'w') as W:
            for line in R:
                v = float(line.replace('\n', '').split()[-1])
                if v < pos_umb:
                    W.write(line)

        INFOMAP = os.path.join(INFOMAP_PATH, INFOMAP_EX)
        os.system(' '.join([INFOMAP, pos_name, os.path.abspath('data'), '--silent']))
        os.system(' '.join([INFOMAP, neg_name, os.path.abspath('data'), '--silent']))

        # Extract the modules center node by max flux criteria
        pos_modules, pos_i = 0, 0
        pos_ides, neg_ides = [], []
        with open(os.path.join('data', os.path.basename(pos_name)+'.tree'), 'r') as file:
            for lines in file.readlines():
                if lines[0] == '#':
                    continue
                mod = int(lines.split()[0].split(':')[0])
                pos_ides.append(int(lines.split()[-1]))

                if mod > pos_modules:
                    pos_i = 1
                    pos_modules = mod
                    pos_c.append(pos[ int(lines.split()[-1]) ].tolist())
                elif mod == pos_modules and pos_i < max_module:
                    pos_i += 1
                    pos_c.append(pos[ int(lines.split()[-1]) ].tolist())
        
        # Add the singles nodes (positives)
        completeNodes(pos_ides, pos_c, pos)
        
        neg_modules, neg_i = 0, 0
        with open(os.path.join('data', os.path.basename(neg_name)+'.tree'), 'r') as file:
            for lines in file.readlines():
                if lines[0] == '#':
                    continue
                mod = int(lines.split()[0].split(':')[0])
                neg_ides.append(int(lines.split()[-1]))

                if mod > neg_modules:
                    neg_i = 1
                    neg_modules = mod
                    neg_c.append(neg[ int(lines.split()[-1]) ].tolist())
                elif mod == neg_modules and neg_i < max_module:
                    neg_i += 1
                    neg_c.append(neg[ int(lines.split()[-1]) ].tolist())
        
        # Add the singles nodes (negatives)
        completeNodes(neg_ides, neg_c, neg)

        os.remove(os.path.join('data', os.path.basename(pos_name)+'.tree'))
        os.remove(os.path.join('data', os.path.basename(neg_name)+'.tree'))
        os.remove(os.path.join('data', os.path.basename(pos_name)))
        os.remove(os.path.join('data', os.path.basename(neg_name)))
        os.remove(os.path.join('data', os.path.basename(pos_name+'t')))
        os.remove(os.path.join('data', os.path.basename(neg_name+'t')))
    elif method == 'c-graph':
        neg_umbral = umbral
        if type(umbral) is not float:
            neg_umbral = umbral[1]
            umbral = umbral[0]

        print ('Making graphs with (positive: {:.6}, negative: {:.6}) threshold'.format(umbral, neg_umbral), 'and compact algorithm')
        bar = MyBar('c-graph', max=len(pos)+len(neg))
        
        pos = pos.to_numpy().tolist()
        pos = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], pos)], dtype=np.float32)
        G = nx.Graph()

        for i in range(pos.shape[0]):
            tmp_v = pos[i].reshape(1, -1)
            if method_distance == 'euclidea':
                tmp_v = np.sqrt(((pos - tmp_v)**2).sum(axis=-1))
            elif method_distance == 'cosine':
                n1 = np.sqrt((tmp_v**2).sum(axis=-1))
                n2 = np.sqrt((pos**2).sum(axis=-1))
                tmp_v = (tmp_v*pos).sum(axis=-1)
                tmp_v = tmp_v/(n1*n2 + eps)
            tmp_v  = tmp_v.reshape(-1)
            ar_pos = -1
            for j in range(tmp_v.shape[0]):
                if j == i or tmp_v[j] > umbral:
                    continue
                if ar_pos < 0:
                    ar_pos = j
                elif tmp_v[j] < tmp_v[ar_pos]:
                    ar_pos = j
            if ar_pos >= 0:
                G.add_edge(i,j, weight=tmp_v[j])
            bar.next()

        accum, max_len_module = 0, 1
        for c in nx.connected_components(G):
            my_node, ac = [], 0
            for n in c:
                ac += 1
                valor, s = 0, 0
                for _, _, d in G.edges(n, data=True):
                    valor += d['weight']
                    s += 1
                valor /= s if s > 0 else 1
                my_node.append((valor, n))
            accum += 1 if ac <= max_len_module else 0
            my_node.sort(reverse=True)
            my_node = my_node[:-min(max_module, len(my_node))]
            for _, i in my_node:
                pos_c.append(pos[i].tolist())
        del G 
        del pos
        
        neg = neg.to_numpy().tolist()
        neg = np.array([i for i in map(lambda x: [float(v) for v in x[0].split()], neg)], dtype=np.float32)
        G = nx.Graph()

        for i in range(neg.shape[0]):
            tmp_v = neg[i].reshape(1, -1)
            if method_distance == 'euclidea':
                tmp_v = np.sqrt(((neg - tmp_v)**2).sum(axis=-1))
            elif method_distance == 'cosine':
                n1 = np.sqrt((tmp_v**2).sum(axis=-1))
                n2 = np.sqrt((neg**2).sum(axis=-1))
                tmp_v = (tmp_v*neg).sum(axis=-1)
                tmp_v = tmp_v/(n1*n2 + eps)
            tmp_v  = tmp_v.reshape(-1)
            ar_neg = -1
            for j in range(tmp_v.shape[0]):
                if j == i or tmp_v[j] > neg_umbral:
                    continue
                if ar_neg < 0:
                    ar_neg = j
                elif tmp_v[j] < tmp_v[ar_neg]:
                    ar_neg = j
            if ar_neg >= 0:
                G.add_edge(i,j, weight=tmp_v[j])
            bar.next()

        for c in nx.connected_components(G):
            my_node, ac = [], 0
            for n in c:
                ac += 1
                valor, s = 0, 0
                for _, _, d in G.edges(n, data=True):
                    valor += d['weight']
                    s += 1
                valor /= s if s > 0 else 1
                my_node.append((valor, n))
            accum += 1 if ac <= max_len_module else 0
            my_node.sort(reverse=True)
            my_node = my_node[:-min(max_module, len(my_node))]
            for _, i in my_node:
                neg_c.append(neg[i].tolist())
        del G 
        del neg        
        bar.finish()
        print ('Total of components less than {}: {}'.format(max_len_module, accum))

    print ('Total of positive centers:', len(pos_c))
    print ('Total of negative centers:', len(neg_c))
    with open(os.path.join('data', 'pos_center.txt'), 'w') as file:
        for l in pos_c:
            file.write(' '.join([str(v) for v in l]) + '\n')
    with open(os.path.join('data', 'neg_center.txt'), 'w') as file:
        for l in neg_c:
            file.write(' '.join([str(v) for v in l]) + '\n')

def _findMyRandom(data, N):
    p   = random.randint(0,len(data)-N)
    dt_ = data.iloc[p:p+N].to_numpy().tolist()
    dt_ = [v[0] for v in dt_]
    return dt_

def _findMyCloser(datu, vector, N, distance='euclidea', eps=1e-7):
    if distance not in ['euclidea', 'cosine']:
        print ('ERROR::DiSTANCE', distance, 'not in', ' '.join(['euclidea', 'cosine']))
        return

    vects = datu.to_numpy().tolist()
    vects = [[float(s) for s in x[0].split()] for x in vects]
    vects = np.array(vects, dtype=np.float32)

    vector = np.array([float(s) for s in vector.split()], dtype=np.float32).reshape(1,-1)

    if distance == 'euclidea':
        vects = (vects - vector)**2
        vects = np.sqrt(vects.sum(axis=1))
    elif distance == 'cosine':
        n1 = np.sqrt((vector**2).sum(axis=1))
        n2 = np.sqrt((vects**2).sum(axis=1))
        vects = (vects*vector).sum(axis=1) / (n1*n2 + eps)
    ides = []
    for i in range(vects.shape[0]):
        ides.append((vects[i], i))
    ides.sort()
    sol, N = [], min(N, len(ides), len(datu))
    for i in range(N):
        sol.append(ides[i][1])
    dt_ = datu.iloc[sol].to_numpy().tolist()
    dt_ = [v[0] for v in dt_]
    return dt_

def _pikMeRandom(arr, K):
    pos:int = 0
    for i in range(K): 
        pos = random.randint(pos, len(arr)-K+i)
        yield arr[ pos ]

def _pikMeCloser(arr, vec, K, distance='euclidea', eps=1e-7):
    if distance not in ['euclidea', 'cosine']:
        print ('ERROR::DiSTANCE', distance, 'not in', ' '.join(['euclidea', 'cosine']))
        return
    vec  = np.array([float(v) for v in vec.split()]).reshape(1,-1)
    vecs = np.array([[float(v) for v in x.split()] for x in arr])

    if distance == 'euclidea':
        vecs = np.sqrt(((vecs - vec)**2).sum(axis=-1))
    elif distance == 'cosine':
        vecs = (vec*vecs).sum(axis=-1) / ( np.sqrt((vec*vec).sum()) * np.sqrt((vecs*vecs).sum(axis=-1)) + eps)
    
    vecs = [(vecs[i], i) for i in range(vecs.shape[0])]
    vecs.sort()
    vecs = vecs[:K]
    vecs = [arr[i] for _,i in vecs]
    return vecs
    

def makeSiamData(data_path:str, K, M, ref_folder='data', humor_label='is_humor', distance='euclidea'):
    data = pd.read_csv(data_path)
    pos  = data.query(humor_label+' == 1').drop([humor_label, 'id', 'humor_rating'], axis=1)
    neg  = data.query(humor_label+' == 0').drop([humor_label, 'id', 'humor_rating'], axis=1)

    old_data,new_data, header = [], [], []
    for s in data.columns:
        old_data.append([])
        header.append(s)
    
    pos_centers, neg_centers = [], []
    if os.path.isfile('data/neg_center.txt'):
        with open('data/neg_center.txt', 'r') as file:
            for line in file.readlines():
                neg_centers.append(line.replace('\n', ''))
    if os.path.isfile('data/pos_center.txt'):
        with open('data/pos_center.txt', 'r') as file:
            for line in file.readlines():
                pos_centers.append(line.replace('\n', ''))
    # pos_centers = pos_centers[:min(K, len(pos_centers))]
    # neg_centers = neg_centers[:min(K, len(neg_centers))]
    
    print ('# Making Siam data from', colorizar(os.path.basename(data_path)))
    bar = MyBar('data', max=len(data))
    for i in range(len(data)):
        dt, da, lab = None, None, None
        if int(data.loc[i, humor_label]) == 0:
            dt = list(_pikMeRandom(neg_centers, K))
            da = _pikMeCloser(pos_centers, data.loc[i, 'vecs'], M, distance=distance)
        else:
            dt = list(_pikMeRandom(pos_centers, K))
            da = _pikMeCloser(neg_centers, data.loc[i, 'vecs'], M, distance=distance)

        for v in dt:
            new_data.append(v)
            for h,p in zip(header, old_data):
                if h == humor_label:
                    p.append('0')
                else:
                    p.append(data.loc[i, h])
        for v in da:
            new_data.append(v)
            for h,p in zip(header, old_data):
                if h == humor_label:
                    p.append('1')
                else:
                    p.append(data.loc[i, h])
        bar.next()
    header.append('vecs_siam')
    bar.finish()
    del data 

    old_data.append(new_data)
    data = pd.concat([pd.Series(s) for s in old_data], axis=1)
    
    new_path = os.path.join('data', 'Siam'+os.path.basename(data_path))
    data.to_csv(new_path, index=None, header=header)
    return new_path

def makeSiam_ZData(data_path:str, model, ref_folder='data', batch=16):
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
            texts = data.loc[i:end, 'vecs'].to_numpy().tolist()
            texts = [np.array(v, dtype=np.float32) for v in map(lambda x: [float(s) for s in x.split()], texts)]
            texts = [ np.concatenate([v.reshape(1,-1)]*int(pos_size+neg_size), axis=0) for v in texts]

            texts = np.concatenate(texts, axis=0)
            vec   = np.concatenate([vectors_np]*int(end-i+1), axis=0)
            in_ve = np.concatenate([texts, vec], axis=1)
            
            y_hat, y_v = model(torch.from_numpy(in_ve))
            y_hat = y_hat.reshape(-1,int(pos_size+neg_size)).numpy() #mirar esto

            new_x.append(y_hat)
            bar.next()
    bar.finish()
    new_x = np.concatenate(new_x, axis=0).tolist()
    new_x = [s for s in map(lambda x: ' '.join([str(v) for v in x]), new_x)]
    data.drop(['vecs'], axis=1, inplace=True)

    new_head = [s for s in data.columns] + ['vecs']
    data = pd.concat([data, pd.Series(new_x)], axis=1)
    data.to_csv(new_name, index=None, header=new_head)
    return new_name, pos_size, neg_size

def convert2EncoderVec(data_name:str, model, loader, save_pickle=False, save_as_numpy=False):
    model.eval()
    IDs, YC, YV, X = [], [], [], []

    new_name = os.path.join('data', data_name+'.csv' if not save_pickle else data_name+'.pkl')

    print ('# Creating', colorizar(os.path.basename(new_name)))
    bar = MyBar('change', max=len(loader))

    cpu0 = torch.device("cpu")
    with torch.no_grad():
        for data in loader:
            x = model(data['x'], ret_vec=True).to(device=cpu0).numpy()
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

    if save_as_numpy:
        X_t = [v for v in map(lambda x: [float(s) for s in x.split()], X)]
        X_t = np.array(X_t, dtype=np.float32)
        np.save(os.path.join('data', data_name+'.npy'), X_t)
        del X_t
        
        if len(IDs) > 0:
            ids_t = np.array([int(i) for i in IDs], dtype=np.int32)
            np.save(os.path.join('data', data_name+'_id.npy'), ids_t)
            del ids_t

    conca, n_head = [], []
    if len(IDs) > 0:
        conca.append(pd.Series(IDs))
        n_head.append('id')
        del IDs
    if len(YC) > 0:
        conca.append(pd.Series(YC))
        n_head.append('is_humor')
        del YC
    if len(YV) > 0:
        conca.append(pd.Series(YV))
        n_head.append('humor_rating')
        del YV
    conca.append(pd.Series(X))
    n_head.append('vecs')
    del X

    data = pd.concat(conca, axis=1)
    if save_pickle:
        data.to_pickle(new_name)
    else:
        data.to_csv(new_name, index=None, header=n_head)
    return new_name

def _eval(y_test, y_predicted):
	# precision, recall, fscore, _ = score(y_test, y_predicted)
	# print('\n     {0}   {1}'.format("0","1"))
	# print('P: {}'.format(precision))
	# print('R: {}'.format(recall))
	# print('F: {}'.format(fscore))

	mprecision, mrecall, mfscore, _ = score(y_test, y_predicted, average='macro')
	print(headerizar('# MACRO-AVG'))
	print('P: {} R: {} F: {}'.format(mprecision, mrecall, mfscore))

def predictManual(data_path:str, N_pos, N_neg, save_name='prediction_manual', shost_compare=False):
    save_name = os.path.join('preds', save_name+'.csv')
    data = pd.read_csv(data_path)

    O, RV = [],[]
    print ('# Manual predictions to', colorizar(os.path.basename(data_path)))
    for i in range(len(data)):
        vec = np.array([float(s) for s in data.loc[i, 'vecs'].split()], dtype=np.float32)
        assert ( (N_pos+N_neg) == vec.shape[0] )
        v1, v2 = vec[:N_pos].min(), vec[N_pos:].min()
        val = '1' if v1 < v2 else '0'
        
        if shost_compare:
            RV.append(data.loc[i, 'is_humor'])
        O.append(val)
    
    if shost_compare:
        RV, O = [int(s) for s in RV], [int(s) for s in O]
        _eval(RV, O)
    else:
        data.drop(['vecs'], axis=1, inplace=True)
        header = list(data.columns) + ['is_humor']
        data = pd.concat([data, pd.Series(O)], axis=1)
        data.to_csv(save_name, header=header, index=None)
        print ('Predictions were saved in', colorizar(save_name))