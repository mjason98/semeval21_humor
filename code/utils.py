'''
	This is a set of functions and classes to help the main prosses but they are pressindible.
'''
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import  Bar
import random

from sklearn.manifold import TSNE 
import collections # reducer
import os # reducer
#import re # reducer
import pickle

class MyBar(Bar):
	empty_fill = '.'
	fill = '='
	bar_prefix = ' ['
	bar_suffix = '] '
	value = 0
	hide_cursor = False
	width = 20
	suffix = '%(percent).1f%% - %(value).5f'
	
	def next(self, _v=None):
		if _v is not None:
			self.value = _v
		super(MyBar, self).next()

def colorizar(text):
	return '\033[91m' + text + '\033[0m'
def headerizar(text):
	return '\033[1m' + text + '\033[0m'


def getMyDict():
	return {'<emogy>':1, '<hashtag>':2, '<url>':3, '<risa>':4, '<signo>':5,
			'<ask>':6, '<phoria>':7, '<diag>':8, '<number>':9, '<date>':10,
			'<sent>':11, '<user>':12, '<frase>':13 }

def generate_dictionary_from_embedding(filename, dictionary, ret=True, logs=True, norm=False, message_logs=''):
	if logs:
		print ('# Loading:', colorizar(os.path.basename(filename)), message_logs)
	x = []
	band, l = False, 0

	mean, var, T = 0, 0, 0
	with open(filename, 'r', encoding='utf-8') as file:
		for ide, line in enumerate(file):
			li = line.split()

			if len(li) <= 2:
				print('#WARNING::', line, 'interpreted as head')
				continue
				
			if not band:
				x.append([0 for _ in range(len(li)-1)])
				my_d = getMyDict()
				l = len(my_d)
				for val in my_d:
					x.append([random.random() for _ in range(len(li)-1)])
					dictionary.update({val.lower(): my_d[val] })
				band = True

			a = [float(i) for i in li[1:]]
			x.append(a)
			
			mean += np.array(a, dtype=np.float32)
			var  += np.array(a, dtype=np.float32)**2
			T += 1

			dictionary.update({li[0].lower(): ide + l + 1})
	var  /= float(T)
	mean /= float(T)
	var -= mean ** 2
	var  = np.sqrt(var)
	mean = mean.reshape(1,mean.shape[0])
	var = var.reshape(1,var.shape[0])
	
	if ret:
		sol = np.array(x, np.float32)
		if norm:
			sol = (sol - mean) / var
		return sol

class TorchBoard(object):
	'''
		This is a naive board to plot the training and the evaluation phase
		using matplotlib
	'''
	def __init__(self):
		self.dict = {}
		self.labels = ['train', 'test']
		self.future_updt = True
		self.best_funct = None
		self.setFunct( max )
		self.best   = [None, None]
		self.best_p = [0, 0]

	def setFunct(self, fun):
		'''
			Set the 'best' function to calculate the best point in the plot

			fun:
				a function pointer
		'''
		self.best_funct = fun

	def update(self, label, value, getBest=False):
		'''
		add a anotation
		label: str
			most be in ['train', 'test']
		value: number
		getBest: bool
			this make the funtion to return True if the value parameter given is the 'best' at the moment
			The 'best' is a function which calculate if the new value is in a sence better than the olders one, this is setted with the setFunct() method 
		'''
		if self.future_updt == False:
			return
		if label not in self.labels:
			print ('WARNING: the label {} its not in {}, the board will not be updated.'.format(
				label, self.labels))
			self.future_updt = False
			return
		pk = 1
		if label == 'train':
			pk = 0

		if self.dict.get(label) == None:
			self.dict.update({label:[value]})
		else:
			self.dict[label].append(value)

		yo = False
		if self.best[pk] is None:
			yo = True
			self.best[pk] = value
		else:
			self.best[pk] = self.best_funct(self.best[pk], value)
			yo = self.best[pk] == value
		if yo:
			self.best_p[pk] = len(self.dict[label]) - 1
		if getBest:
			return yo

	def show(self, saveroute, plot_smood=False):
		'''
			Save the plot

			saverroute: str
				path to save the plot
			plt_smood: bool
				If is True, plot a dotted curve repressentig the smood real curve.
		'''
		fig , axes = plt.subplots()
		for i,l in enumerate(self.dict):
			y = self.dict[l]
			if len(y) <= 1:
				continue
			lab = str(self.best[i])
			if len(lab) > 7:
				lab = lab[:7]
			axes.plot(range(len(y)), y, label=l + ' ' + lab)
			axes.scatter([self.best_p[i]], [self.best[i]])

			if plot_smood:
				w = 3
				y_hat = [ np.array(y[max(i-w,0):min(len(y),i+w)]).mean() for i in range(len(y))]
				axes.plot(range(len(y)), y_hat, ls='--', color='gray')

		fig.legend()
		fig.savefig(saveroute)
		del axes
		del fig

def reduced(oldFile, newFile, vocab):
	'''
	Reduce the oldFile embedding to a new one in which the words match de vocab variable

	inputs:
		oldFile: str 
			old embedding file
		newFile: str 
			path to the new embedding path
		vocab: dict 
			vocabulary to filter the older file
	'''
	print ('# Turning', colorizar(oldFile), 'into', colorizar(newFile))
	
	file = open (newFile, 'w')
	with open(oldFile, 'r', encoding='utf-8') as oldf:
		for line in oldf.readlines():
			l = line.split()
			if len(l) <= 2:
				continue
			word = l[0].lower()
			if vocab.get(word, 0) != 0:
				file.write(line)
				vocab.pop(word)
	file.close()
	print('# Done!')


def makeVocabFromData(filepath):
	'''
	Make a vocabulary from a file

	input:
		filepath: str
			This file is splited with a space separator (' ') and the words are returned

	output:
		vocabulary: dict
	'''
	c = None
	with open(filepath, 'r', encoding='utf-8') as f:
		line = f.read().replace('\n', ' ')
		c = collections.Counter(line.split())

	return dict([(i, 5) for i in sorted(c, reverse=True)])

def projectData2D(data_path:str, save_name='2Data', use_centers=False):
	'''
		Project the vetors in 2d plot

		data_path:str most be a cvs file
	'''
	data = pd.read_csv(data_path)

	np_data = data.drop(['is_humor','humor_rating', 'id'], axis=1).to_numpy().tolist()
	np_data = [i for i in map(lambda x: [float(v) for v in x[0].split()], np_data)]
	np_data = np.array(np_data, dtype=np.float32)

	L = []
	if use_centers:
		P = ['neg_center.txt', 'pos_center.txt']
		for l in P:
			with open(os.path.join('data', l), 'r') as file:
				lines = file.readlines()
				lines = np.array([[float(v) for v in x.split()] for x in lines], dtype=np.float32)
				L.append(lines)
	L = np.concatenate(L, axis=0)
	# axes.scatter(lines[:,0], lines[:,1], c='r')

	print ('Projecting', colorizar(os.path.basename(data_path)), 'in 2d vectors')
	np_data = np.concatenate([np_data, L], axis=0)
	L = L.shape[0]

	X_embb = TSNE(n_components=2).fit_transform(np_data)
	#X_embb = PCA(n_components=2, svd_solver='full').fit_transform(np_data)
	#X_embb = TruncatedSVD(n_components=2).fit_transform(np_data)
	print ('Done!')
	del np_data
	
	D_1, D_2 = [], []
	for i in range(len(data)):
		if int(data.loc[i, 'is_humor']) == 0:
			D_1.append([X_embb[i,0], X_embb[i,1]])
		else:
			D_2.append([X_embb[i,0], X_embb[i,1]])
	X_embb = X_embb[-L:]

	D_1, D_2 = np.array(D_1), np.array(D_2)
	fig , axes = plt.subplots()
	axes.scatter(D_1[:,0], D_1[:,1], label='neg', c='gray')
	axes.scatter(D_2[:,0], D_2[:,1], label='pos', c='b')

	axes.scatter(X_embb[:,0], X_embb[:,1], label='centers', c='r')

	fig.legend()
	fig.savefig(os.path.join('pics', save_name+'.png'))
	# plt.show()
	del fig
	del axes
	del X_embb
	
