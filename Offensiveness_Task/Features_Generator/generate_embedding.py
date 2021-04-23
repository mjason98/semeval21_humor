#%%
import glob
import numpy as np
import re
import pandas as pd

path = '../../data/affective_resources/'
        # home/nitro/projects/Semeval/data/features/Figurative-Datasets/datosIronyDetection/recursos
dic = set()
addrs = np.array(glob.glob(path + '*.txt') + glob.glob(path + 'binary/*.txt') )
phl = '../../data/affective_resources/hurtlex_EN_conservative.tsv'

for adr in addrs:
    print(adr)
    with open(adr, 'r') as file:
        for i in file.readlines():
            for s in ['\t', '\n']:
                i = i.split(s)[0]

            dic.add(i)

with open(phl, 'r') as file:
    for i in file.readlines():
        x = i.split('\t')
        dic.add(x[2])           

index = {}
for i, id in zip(dic, range(len(dic))):
    index[i] = id

embedding_matrix = np.zeros((len(dic)+1, 1))

# Binary Features
addrs = np.array(glob.glob(path + 'binary/*.txt') )

with open(addrs[0], 'r') as file:
    for i in file.readlines():
        for s in ['\t', '\n']:
            i = i.split(s)[0]
        embedding_matrix[index[i]][0] = 1

for adr in addrs[1:]:
    print(adr)
    feature = np.zeros((len(dic)+1, 1))
    with open(adr, 'r') as file:
        for i in file.readlines():
            for s in ['\t', '\n']:
                i = i.split(s)[0]

            feature[index[i]][0] = 1
    embedding_matrix = np.concatenate([embedding_matrix, feature], axis=1)

# Continuous Feeatures

addrs = np.array(glob.glob(path + '*.txt') )

for adr in addrs:
    print(adr)
    feature = np.zeros((len(dic)+1, 3))
    with open(adr, 'r') as file:
        for i in file.readlines():
            word, coefs = i.split(maxsplit=1)   
            coefs = np.fromstring(coefs, 'f', sep=' ')
            feature[index[word]] = np.array(coefs)
        
    embedding_matrix = np.concatenate([embedding_matrix, feature], axis=1)

#Adding Hurlex


# PS:  Negative stereotypes ethnic slurs
# RCI: locations and demonyms
# PA:  professions and occupations 
# DDF: physical disabilities and diversity
# DDP  cognitive disabilities and diversity
# DMC: moral and behavioral defects
# IS:  words related to social and economic disadvantage
# OR:  plants
# AN:  animals 
# ASM: male genitalia
# ASF: female genitalia
# PR:  words related to prostitution
# OM:  words related to homosexuality
# QAS: with potential negative connotations
# CDS: derogatory words
# RE:  felonies and words related to crime and immoral behavior
# SVP: words related to the seven deadly sins of the Christian tradition
categories =[ i.lower() for i in ['PS','RCI', 'PA', 'DDF', 'DDP', 'DMC', 'IS', 'OR', 'AN', 'ASM', 'ASF', 'PR', 'OM', 'QAS', 'CDS', 'RE','SVP']]
feature = np.zeros((len(dic)+1, 1))

with open(phl, 'r') as file:
    for i in file.readlines():
        x = i.split('\t')
        feature[index[x[2]]] = categories.index(x[0])+1

import keras as K

feature = K.utils.to_categorical(feature)
embedding_matrix = np.concatenate([embedding_matrix, feature], axis=1)

# cat = file.to_numpy()

#%%
# dic = np.array(dic)
np.save('affective_dic', list(dic))
np.save('affective_embedding', embedding_matrix)

# %%
