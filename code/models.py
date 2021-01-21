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
OFFLINE_PATH = ""
ONLINE_NAME  = ""
TOKENIZER_PRE = None
TRANS_MODEL   = None

def offline(band: bool):
	global OFFLINE
	OFFLINE = band

def setOfflinePath(path:str):
	global OFFLINE_PATH
	OFFLINE_PATH = path
def setOnlineName(name:str):
	global ONLINE_NAME
	ONLINE_NAME = name

def make_bert_pretrained_model(mod_only=False):
	'''
		This function return (tokenizer, model)
	'''
	tokenizer, model = None, None

	if not OFFLINE:
		tokenizer = AutoTokenizer.from_pretrained(ONLINE_NAME)
		model = AutoModel.from_pretrained(ONLINE_NAME)
	else:
		tokenizer = AutoTokenizer.from_pretrained(OFFLINE_PATH)
		model = AutoModel.from_pretrained(OFFLINE_PATH)

	if mod_only:
		return model 
	else:
		return tokenizer, model

def makeTrain_and_ValData(data_path:str, percent=10, class_label=None):
	'''
		class_lable: str The label to split, the humor column with values ['0', '1']
	'''
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

	
	train_data, eval_data = None, None
	if class_label is None:
		percent = (len(data) * percent) // 100
		ides = [i for i in range(len(data))]
		random.shuffle(ides)

		train_data = data.drop(ides[:percent])
		eval_data  = data.drop(ides[percent:])
	else:
		pos  = list(data.query(class_label+' == 1').index)
		neg  = list(data.query(class_label+' == 0').index)
		random.shuffle(pos)
		random.shuffle(neg)

		p1,p2 = (len(pos) * percent) // 100, (len(neg) * percent) // 100
		indes_t, indes_e = pos[:p1] + neg[:p2], pos[p1:] + neg[p2:]

		train_data = data.drop(indes_t)
		eval_data  = data.drop(indes_e)

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
			# vects = out[0][:,-1].numpy() # last
			# vects = max_pool_op(out[0]).numpy() # Maxpool
			vects = F.normalize(out[0].sum(dim=1), dim=-1).numpy() # Add and Norm
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

class RawDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		ids  = self.data_frame.loc[idx, 'id']
		sent  = self.data_frame.loc[idx, 'text']
		
		try:
			value = self.data_frame.loc[idx, 'is_humor']
			regv  = self.data_frame.loc[idx, 'humor_rating'] if int(value) != 0 else 0.
		except:
			value, regv = 0, 0.

		sample = {'x': sent, 'y': value, 'v':regv, 'id':ids}
		return sample

class VecsDataset(Dataset):
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
		
		sent  = self.transF(self.data_frame.loc[idx, 'x'])
		value = self.data_frame.loc[idx, 'y_c']
		regv  = self.data_frame.loc[idx, 'y_v']

		sample = {'x': sent, 'y': value, 'v':regv}
		return sample

class SiamDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def transF(self, text1, text2):
		text1 = [ float(i) for i in text1.split() ]
		text2 = [ float(i) for i in text2.split() ]
		text = text1 + text2
		return torch.Tensor(text).float()

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
			
		sent1  = self.transF(self.data_frame.loc[idx, 'vecs'], self.data_frame.loc[idx, 'vecs_siam'])
		value = self.data_frame.loc[idx, 'is_humor']

		sample = {'x': sent1, 'v': 0, 'y': value}
		return sample

class ZODataset(Dataset):
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

		ids  = self.data_frame.loc[idx, 'id']
		sent  = self.transF(self.data_frame.loc[idx, 'vecs'])
		
		try:
			value = self.data_frame.loc[idx, 'is_humor']
			regv  = self.data_frame.loc[idx, 'humor_rating'] if int(value) != 0 else 0.
		except:
			value, regv = 0, 0.

		sample = {'x': sent, 'y': value, 'v':regv, 'id':ids}
		return sample

def makeDataSet_ZO(csv_path:str, batch, shuffle=True):
	data   =  ZODataset(csv_path)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=shuffle, num_workers=4,
							drop_last=False)
	return data, loader

def makeDataSet_Raw(csv_path:str, batch, shuffle=True):
	data   =  RawDataset(csv_path)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=shuffle, num_workers=4,
							drop_last=False)
	return data, loader

def makeDataSet_Vecs(csv_path:str, batch, shuffle=True):
	data   =  VecsDataset(csv_path)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=shuffle, num_workers=4,
							drop_last=False)
	return data, loader

def makeDataSet_Siam(csv_path:str, batch, shuffle=True):
	data   =  SiamDataset(csv_path)
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
# Added Memory Augmented Network
class MAN(nn.Module):
	"""docstring for MAN"""
	def __init__(self, in_size, hidd_size, memory_size=50, alpha_sig=-0.4, gamma=0.3):
		super(MAN, self).__init__()
		self.memory_size = memory_size
		self.memory_vector_size = hidd_size

		# self.controler = nn.LSTMCell(in_size, hidd_size)
		self.controler = nn.Sequential(nn.Linear(in_size, hidd_size), nn.LeakyReLU())

		self.M = np.zeros((self.memory_size, self.memory_vector_size), dtype=np.float32)
		self.gate = 1. / (1. + math.exp(-alpha_sig))
		self.gamma = gamma

		self.w_u      = None
		self.prev_w_r = None

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.to(device=self.device)
	
	def init_Batch(self, batch):
		self.w_u      = np.zeros((batch, self.memory_size))
		self.prev_w_r = np.zeros((batch, self.memory_size))

	def read_head(self, K, prev_M_t):
		inner_dot = K @ prev_M_t.T
		k_sq = K.pow(2).sum(dim=-1, keepdim=True).sqrt()
		m_sq = prev_M_t.pow(2).sum(dim=-1, keepdim=True).sqrt()
		norm_dot = k_sq @ m_sq.T
		coss_sim = inner_dot / (norm_dot + 1e-8)
		coss_sim = F.softmax(coss_sim, dim=-1)
		return coss_sim

	def write(self, k, w_r, w_u, prev_r, prev_M_t):
		with torch.no_grad():
			prev_w_u = torch.from_numpy(w_u).detach().to(device=self.device)
			prev_w_r = torch.from_numpy(prev_r).detach().to(device=self.device)
			key      = k.detach()
			w_lu = torch.argmin(prev_w_u, dim=-1).long()
			w_lu = (1. - self.gate) * F.one_hot(w_lu, num_classes=self.memory_size).float()
			w_w  = self.gate*prev_w_r + w_lu

			# usar prev_M_T para normalizar si da error
			k_w  = torch.einsum('bi,bj->bij', w_w, key).sum(dim=0)
			k_w  = F.normalize(prev_M_t + k_w, dim=-1).cpu().numpy()
			self.M[:,:] = k_w[:,:]

			prev_r[:,:] = w_r.detach().numpy()[:,:]
			new_u    = F.normalize(self.gamma*prev_w_u + w_r + w_w, dim=-1).numpy()
			w_u[:,:]   = new_u[:,:]

	def forward(self, X):
		if self.w_u is None or self.w_u.shape[0] != X.shape[0]:
			self.init_Batch(X.shape[0])

		M = torch.from_numpy(self.M).detach().to(device=self.device)
		# Read from the memory
		h = self.controler(X)
		w_r = self.read_head(h, M)
		read_i = w_r @ M
		out = torch.cat([h, read_i], dim=-1).reshape(X.shape[0], self.memory_vector_size*2)

		# Write to the memory
		self.write(h, w_r, self.w_u, self.prev_w_r, M)

		return out

class MXP(torch.nn.Module):
	def __init__(self):
		super(MXP, self).__init__()
	def forward(self, X):
		seq_len = X.shape[1]
		x_hat = X.permute((0,2,1))
		x_hat = F.max_pool1d(x_hat,seq_len, stride=1)
		x_hat = x_hat.permute((0,2,1))
		return x_hat.squeeze()

class ADDN(torch.nn.Module):
	def __init__(self):
		super(ADDN, self).__init__()
	def forward(self, X):
		return F.normalize(X.sum(dim=1), dim=-1)

class POS(torch.nn.Module):
	def __init__(self, _p = 0):
		super(POS, self).__init__()
		self._p = _p
	def forward(self, X):
		return X[:,self._p]

class ATT(torch.nn.Module):
	def __init__(self, in_size = 768):
		super(ATT, self).__init__()
		self.A = nn.Linear(in_size, 1)
	def forward(self, X):		
		alpha = F.softmax(self.A(X), dim=-1)
		return (alpha * X).sum(dim=1)

class Encod_Model(nn.Module):
	def __init__(self, hidden_size, vec_size, dropout=0.1, use_man=False, mm=250):
		super(Encod_Model, self).__init__()
		self.mid_size = 64
		self.Dense1   = nn.Sequential(nn.Linear(vec_size, hidden_size), nn.LeakyReLU(), nn.Dropout(dropout), 
									  nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(), 
									  nn.Linear(hidden_size//2, self.mid_size), nn.LeakyReLU())
		if use_man:
			self.Task1 = nn.Sequential(MAN(self.mid_size, self.mid_size//2, memory_size=mm), nn.Linear(2*(self.mid_size//2), 2))
		else:
			self.Task1   = nn.Linear(self.mid_size, 2)
		self.Task2   = nn.Linear(self.mid_size, 1)

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
	
	def load_memory(self, path):
		pp = os.path.join(os.path.dirname(path), 'memory_'+os.path.basename(path)+'.npy')
		self.Task1[0].M = np.load(pp)
	def save_memory(self, path):
		pp = os.path.join(os.path.dirname(path), 'memory_'+os.path.basename(path)+'.npy')
		np.save(pp, self.Task1[0].M)

class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""
	def __init__(self, margin=2.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	# ver lo del contrastive con lo del label
	def forward(self, D, label):
		loss_contrastive = torch.mean((1-label) * torch.pow(D, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - D, min=0.0), 2))
		return loss_contrastive

class Siam_Model(nn.Module):
	def __init__(self, hidden_size, vec_size, dropout=0.01):
		super(Siam_Model, self).__init__()
		# self.criterion = nn.CrossEntropyLoss()#weight=torch.Tensor([0.7,0.3]))
		self.criterion1 = ContrastiveLoss(1.0)

		self.Dense1   = nn.Sequential(nn.Linear(vec_size, hidden_size), nn.LeakyReLU(), nn.Dropout(dropout), 
									  nn.Linear(hidden_size, hidden_size//2), nn.LeakyReLU(), 
									  nn.Linear(hidden_size//2, hidden_size//4))

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.to(device=self.device)

	def forward(self, X):
		size = X.shape[1] //2
		X1, X2 = X[:,:size].to(device=self.device), X[:,size:].to(device=self.device)

		y1 = self.Dense1(X1)
		y2 = self.Dense1(X2)

		# distance function
		euclidean_distance = F.pairwise_distance(y1, y2)
		return euclidean_distance, 0

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path)

class Z_Model(nn.Module):
	def __init__(self, hsize, vec_size, dropout=0.2, memory_size=250):
		super(Z_Model, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()#weight=torch.Tensor([0.7,0.3]))

		self.hsize = hsize
		self.L1 = MAN(vec_size, hsize, memory_size=memory_size)

		self.siam_blk = nn.Sequential(nn.Linear(hsize, hsize//2), nn.LeakyReLU(), nn.Linear(hsize//2, hsize//4))
		# self.Task1 = nn.Sequential(nn.Linear(hsize*2, hsize), nn.LeakyReLU(), nn.Linear(hsize, 2))
		self.Task1 = nn.Linear(hsize//4, 2)
		self.Task2 = nn.Linear(hsize//4, 1)

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.to(device=self.device)

	def forward(self, X):
		y_t = self.L1(X)

		# y1 = self.siam_blk(y_t[:, :self.hsize])
		# y2 = self.siam_blk(y_t[:, self.hsize:])
		# y_t = y1 - y2

		y1 = self.Task1(y_t)
		y2 = self.Task2(y_t)
		return y1, y2

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 

class MaskedMSELoss(torch.nn.Module):
	def __init__(self):
		super(MaskedMSELoss, self).__init__()
		self.mse = nn.MSELoss(reduction='none')

	def forward(self, y_hat, y, label):
		y_loss = self.mse(y_hat, y).reshape(-1)
		y_mask = label.reshape(-1)
		return (y_loss*y_mask).mean()

class Bencoder_Model(nn.Module):
	def __init__(self, hidden_size, vec_size=768, dropout=0.1, max_length=120, selection='addn', use_man=False, mm=250):
		'''
			selection: ['addn', 'first', 'mxp', 'att']
		'''
		super(Bencoder_Model, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()#weight=torch.Tensor([0.7,0.3]))
		self.criterion2 = MaskedMSELoss()
		self.use_man = use_man
		# self.criterion2 = nn.CrossEntropyLoss(reduction='none')#weight=torch.Tensor([0.7,0.3]))

		self.max_length = max_length
		self.tok, self.bert = make_bert_pretrained_model()
		self.encoder = Encod_Model(hidden_size, vec_size, dropout=dropout, use_man=use_man, mm=mm)
		self.selection = None

		if selection   == 'addn':
			self.selection = ADDN()
		elif selection == 'mxp':
			self.selection = MXP()
		elif selection == 'first':
			self.selection = POS(0)
		elif selection == 'att':
			self.selection = ATT(in_size=vec_size)

		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.to(device=self.device)
		
	def forward(self, X, ret_vec=False):
		ids   = self.tok(X, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)
		out   = self.bert(**ids)
		vects = self.selection(out[0])
		return self.encoder(vects, ret_vec=ret_vec)

	def load(self, path):
		self.load_state_dict(torch.load(path, map_location=self.device))
		if self.use_man:
			self.encoder.load_memory(path)

	def save(self, path):
		torch.save(self.state_dict(), path) 
		if self.use_man:
			self.encoder.save_memory(path)
	
	def makeOptimizer(self, lr=5e-5, decay=2e-5, algorithm='adam', lr_fin=3e-5):
		pars = [{'params':self.encoder.parameters()}]
		mtl, lamda = 1. / len(self.bert.encoder.layer), 0
		
		for l in self.bert.encoder.layer:
			lr_t = (1. - lamda)*lr + lamda*lr_fin
			D = {'params':l.parameters(), 'lr':lr_t}
			pars.append(D)
			lamda += mtl 
		try:
			lr_t = (1. - lamda)*lr + lamda*lr_fin
			D = {'params':self.bert.pooler.parameters(), 'lr':lr_t}
			pars.append(D)
		except:
			print('#Warning: Pooler layer not found')

		if algorithm == 'adam':
			return torch.optim.Adam(pars, lr=lr_fin, weight_decay=decay)
		elif algorithm == 'rms':
			return torch.optim.RMSprop(pars, lr=lr_fin, weight_decay=decay)
	

def makeModels(name:str, size, in_size=768, dpr=0.1, selection='addn', memory_size=None):
	if name == 'encoder':
		return Encod_Model(size, in_size, dropout=dpr)
	elif name == 'bencoder':
		return Bencoder_Model(size, in_size, dropout=dpr, selection=selection, use_man= True if memory_size is not None else False, mm=memory_size)
	elif name == 'siam':
		return Siam_Model(size, in_size, dropout=dpr)
	elif name == 'zmod':
		return Z_Model(size, in_size, memory_size=memory_size if memory_size is not None else 250)
	else:
		print ('ERROR::NAME', name, 'not founded!!')

def trainModels(model, Data_loader, epochs:int, evalData_loader=None, lr=0.1, etha=1., nameu='encoder', optim=None, b_fun=None, smood=False, mtl=True, use_acc=True):
	if epochs <= 0:
		return
	if optim is None:
		optim = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()

	board = TorchBoard()
	if b_fun is not None:
		board.setFunct(b_fun)
	
	for e in range(epochs):
		bar = MyBar('Epoch '+str(e+1)+' '*(int(math.log10(epochs)+1) - int(math.log10(e+1)+1)) , 
					max=len(Data_loader)+(len(evalData_loader if evalData_loader is not None else 0)))
		total_loss, total_acc, dl = 0., 0., 0
		for data in Data_loader:
			optim.zero_grad()
			
			# Multi-Task learning
			y_hat, y_val = model(data['x'])
			y1 = data['y'].to(device=model.device)
			y2 = data['v'].float().to(device=model.device)
			
			if mtl:
				l1 = model.criterion1(y_hat, y1)
				l2 = model.criterion2(y_val, y2, y1)
				loss = etha*l1 + (1. - etha)*l2
			else:
				loss = model.criterion1(y_hat, y1)
			
			loss.backward()
			optim.step()

			with torch.no_grad():
				total_loss += loss.item() * y1.shape[0]
				total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
				dl += y1.shape[0]
			bar.next(total_loss/dl)
		if use_acc:
			res = board.update('train', total_acc/dl, getBest=True)
		else:
			res = board.update('train', total_loss/dl, getBest=True)
		
		# Evaluate the model
		if evalData_loader is not None:
			total_loss, total_acc, dl= 0,0,0
			with torch.no_grad():
				for data in evalData_loader:
					y_hat, y_val = model(data['x'])
					y1 = data['y'].to(device=model.device)
					y2 = data['v'].to(device=model.device)

					if mtl:
						l1 = model.criterion1(y_hat, y1)
						l2 = model.criterion2(y_val, y2, y1)
						loss = etha*l1 + (1. - etha)*l2
					else:
						loss = model.criterion1(y_hat, y1)
					
					total_loss += loss.item() * y1.shape[0]
					total_acc += (y1 == y_hat.argmax(dim=-1)).sum().item()
					dl += y1.shape[0]
					bar.next()
			if use_acc:
				res = board.update('test', total_acc/dl, getBest=True)
			else:
				res = board.update('test', total_loss/dl, getBest=True)
		bar.finish()
		del bar
		
		if res:
			model.save(os.path.join('pts', nameu+'.pt'))
	board.show(os.path.join('pics', nameu+'.png'), plot_smood=smood)

def evaluateModels(model, testData_loader, header=('id', 'is_humor', 'humor_rating'), cleaner=[], name='pred'):
	model.eval()
	
	pred_path = os.path.join('preds', name+'.csv')

	bar = MyBar('test', max=len(testData_loader))
	Ids, lab, val = [], [], []
	
	cpu0 = torch.device("cpu")
	with torch.no_grad():
		for data in testData_loader:
			y_hat, y_val = model(data['x'])
			y_hat, y_val = y_hat.to(device=cpu0), y_val.to(device=cpu0)
			
			y_hat = y_hat.argmax(dim=-1).squeeze()
			y_val = y_val.squeeze() * y_hat
			ids = data['id'].squeeze()
			
			for i in range(ids.shape[0]):
				Ids.append(ids[i].item())
				lab.append(y_hat[i].item())
				val.append(y_val[i].item())
			bar.next()
	bar.finish()
	
	Ids, lab, val = pd.Series(Ids), pd.Series(lab), pd.Series(val)
	data = pd.concat([Ids, lab, val], axis=1)
	del Ids
	del lab
	del val
	data.to_csv(pred_path, index=None, header=header)
	del data
	print ('# Predictions saved in', colorizar(pred_path))
	
	if len(cleaner) > 0:
		data = pd.read_csv(pred_path)
		data.drop(cleaner, axis=1, inplace=True)
		data.to_csv(pred_path, index=None)
		print ('# Cleaned from', ', '.join(cleaner) + '.')
