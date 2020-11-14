from .utils import MyBar, colorizar, TorchBoard

import pandas as pd
import numpy as np
import os
import random
import math

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

OFFLINE = False
OFFLINE_PATH = "/DATA/Mainstorage/Prog/NLP/vinai/bertweet-base"
TOKENIZER_PRE = None
TRANS_MODEL   = None

def offline(band: bool):
	global OFFLINE
	OFFLINE = band

def make_bert_pretrained_model():
	'''
		This function return (tokenizer, model)
	'''
	tokenizer, model = None, None

	if not OFFLINE:
		tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
		model = AutoModel.from_pretrained("vinai/bertweet-base")
	else:
		tokenizer = AutoTokenizer.from_pretrained(OFFLINE_PATH)
		model = AutoModel.from_pretrained(OFFLINE_PATH)

	return tokenizer, model

def makeTrain_and_ValData(data_path:str, percent=10):
	train_path = os.path.join('data', 'train_data.csv')
	eval_path  = os.path.join('data', 'eval_data.csv')

	if os.path.isfile(train_path) and os.path.isfile(eval_path):
		return train_path, eval_path	

	data = pd.read_csv(data_path)	
	mean = [len(data.loc[i, 'text'].split()) for i in range(len(data))]
	var  = [i*i for i in mean]
	mean, var = sum(mean)/len(mean), sum(var)/len(mean)
	var = (var - mean) ** 0.5
	print ('# Mean:', mean, 'std:', var)

	percent = (len(data) * percent) // 100
	ides = [i for i in range(len(data))]
	random.shuffle(ides)

	train_data = data.drop(ides[:percent])
	eval_data  = data.drop(ides[percent:])
	
	train_data.to_csv(train_path, index=None)
	eval_data.to_csv(eval_path, index=None)

	return train_path, eval_path

def load_transformers():
	global TOKENIZER_PRE
	global TRANS_MODEL
	TOKENIZER_PRE, TRANS_MODEL = make_bert_pretrained_model()

def delete_transformers():
	global TOKENIZER_PRE
	global TRANS_MODEL
	del TOKENIZER_PRE
	del TRANS_MODEL

def make_BertRep_from_data(data_path:str, drops:list = ['humor_controversy', 'offense_rating'], 
						   text_field='text', max_length=30, my_batch=32, header_out=('y_c', 'y_v', 'x'),
						   final_drop:list = ['text', 'id']):
	new_data_path = os.path.join('data', 'tr_'+os.path.basename(data_path))
	data = pd.read_csv(data_path)
	data = data.drop(drops, axis=1)
	vetors   = []

	if os.path.isfile(new_data_path):
		return new_data_path

	if TOKENIZER_PRE is None or TRANS_MODEL is None:
		print ('error')
		return 

	print ("# Making vector representations from", colorizar(os.path.basename(data_path)))
	bar = MyBar('data2vec', max=len(data)//my_batch)
	with torch.no_grad():
		for i in range(0,len(data),my_batch):
			end  = min(i+my_batch-1, len(data)-1)
			text = data.loc[i:end, text_field].tolist()

			ids  = TOKENIZER_PRE(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
			out  = TRANS_MODEL(**ids) #, output_hidden_states=True)
			vects = F.normalize(out[0].sum(dim=1), dim=-1).numpy()
			#vects = F.normalize(out.hidden_states[2].sum(dim=1), dim=-1).reshape(-1).numpy()

			for i in range(vects.shape[0]):
				vetors.append(' '.join([str(vects[i,j]) for j in range(vects.shape[1])]))
			bar.next()
	bar.finish()
	
	vetors = pd.Series(vetors)
	data = data.drop(final_drop, axis=1)
	data   = pd.concat([data, vetors], axis=1)

	data.to_csv(new_data_path, index=None, header=header_out)
	return new_data_path

# ================================ DATAS ========================================

class VecsDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def trans(self, text):
		text = [ int(i) for i in text.split() ]
		return torch.Tensor(text).long()

	def transF(self, text):
		text = [ float(i) for i in text.split() ]
		return torch.Tensor(text).float()

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
			
		sent  = self.transF(self.data_frame.loc[idx, 'x'])
		value = self.data_frame.loc[idx, 'y_c']
		regv  = self.data_frame.loc[idx, 'y_v']

		sample = {'x': sent, 'y': value, 'v':regv}
		return sample

def makeDataSet_Vecs(csv_path:str, batch, shuffle=True):
	data   =  VecsDataset(csv_path)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=shuffle, num_workers=4,
							drop_last=False)
	return data, loader

class PredictDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def transF(self, text):
		text = [ float(i) for i in text.split() ]
		return torch.Tensor(text).float()

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
			
		sent = self.transF(self.data_frame.loc[idx, 'x'])
		ids  = self.data_frame.loc[idx, 'id']

		sample = {'x': sent, 'id': ids}
		return sample

def makePredictData(csv_path:str, batch):
	data   =  PredictDataset(csv_path)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=False, num_workers=4,
							drop_last=False)
	return data, loader

# ================================ MODELS ========================================

class Encod_Model(nn.Module):
	def __init__(self, hidden_size, vec_size, dropout=0.2):
		super(Encod_Model, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()#weight=torch.Tensor([0.7,0.3]))
		# self.criterion2 = nn.CrossEntropyLoss(reduction='none')#weight=torch.Tensor([0.7,0.3]))
		# self.nora     = nn.BatchNorm1d(hidden_size*2)

		self.Dense1   = nn.Sequential(nn.Linear(vec_size, hidden_size), nn.LeakyReLU(), #nn.Dropout(dropout), 
		                              nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(), 
									  nn.Linear(hidden_size//2, hidden_size//4), nn.LeakyReLU())
		self.Task1   = nn.Linear(hidden_size//4, 2)
		self.Task2   = nn.Linear(hidden_size//4, 1)
		
	def forward(self, X, ret_vec=False):
		y_hat = self.Dense1(X)
		if ret_vec:
			return y_hat
		y1 = self.Task1(y_hat).squeeze()
		y2 = self.Task2(y_hat).squeeze()
		return y1, y2


	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 
	
def makeModels(name:str, size, in_size=768, dpr=0.2):
	if name == 'encoder':
		return Encod_Model(size, in_size, dropout=dpr)
	else:
		print ('ERROR::NAME', name, 'not founded!!')

def trainModels(model, Data_loader, epochs:int, evalData_loader=None, lr=0.1):
	# eta   = 0.75
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()

	board = TorchBoard()
	for e in range(epochs):
		bar = MyBar('Epoch '+str(e+1)+' '*(int(math.log10(epochs)+1) - int(math.log10(e+1)+1)) , 
					max=len(Data_loader)+(len(evalData_loader if evalData_loader is not None else 0)))
		total_loss, total_acc, dl = 0., 0., 0
		for data in Data_loader:
			optim.zero_grad()
			
			# Multi-Task learning
			y_hat, y_val = model(data['x'])
			y1, y2  = data['y'], data['v']

			loss = model.criterion1(y_hat, y1)
			loss.backward()
			optim.step()

			with torch.no_grad():
				total_loss += loss.item()
				total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
				dl += y1.shape[0]
			bar.next(total_loss/dl)
		res = board.update('train', total_acc/dl, getBest=True)
		
		# Evaluate the model
		if evalData_loader is not None:
			total_acc, dl= 0,0
			with torch.no_grad():
				for data in evalData_loader:
					y_hat, y_val = model(data['x'])
					y1, y2  = data['y'], data['v']

					total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
					dl += y1.shape[0]
					bar.next()
			res = board.update('test', total_acc/dl, getBest=True)
		bar.finish()
		del bar
		
		if res:
			model.save(os.path.join('pts', 'encoder.pt'))
	board.show(os.path.join('pics', 'encoder.png'))

def evaluateModels(model, testData_loader, header=('id', 'is_humor', 'humor_rating')):
	model.eval()
	
	pred_path = os.path.join('data', 'pred.csv')

	bar = MyBar('test', max=len(testData_loader))
	Ids, lab, val = [], [], []
	
	with torch.no_grad():
		for data in testData_loader:
			y_hat, y_val = model(data['x'])
			ids = data['id'].squeeze()
			y_hat = y_hat.argmax(dim=-1).squeeze()
			y_val = y_val.squeeze()
			
			bar.next()
			for i in range(ids.shape[0]):
				Ids.append(ids[i].item())
				lab.append(y_hat[i].item())
				val.append(y_val[i].item())
	bar.finish()
	
	Ids, lab, val = pd.Series(Ids), pd.Series(lab), pd.Series(val)
	data = pd.concat([Ids, lab, val], axis=1)
	del Ids
	del lab
	del val
	data.to_csv(pred_path, index=None, header=header)
	del data
	print ('# Predictions saved in', colorizar(os.path.basename(pred_path)))
