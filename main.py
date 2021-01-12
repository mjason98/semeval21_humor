import os
import sys
import torch
import argparse
import numpy as np
import random

from code.models import makeTrain_and_ValData
from code.models import offline, load_transformers, delete_transformers
from code.models import makeModels, trainModels, makeDataSet_Vecs
from code.models import evaluateModels, makePredictData
from code.models import makeDataSet_Siam, makeDataSet_Raw
from code.models import setOfflinePath, setOnlineName
from code.utils  import projectData2D
from code.siam   import makeSiam_ZData, predictManual, setInfomapData
from code.siam   import findCenter_and_Limits, makeSiamData, convert2EncoderVec

# =============================================
DATA_PATH       = 'data/train.csv'
EVAL_DATA_PATH  = 'data/dev.csv'
TEST_DATA_PATH  = 'data/public_test.csv'
#==============================================

BATCH  = 64
EPOCHS = 20
LR1	   = 5e-5
LR2	   = 3e-5
DPR    = 0.0
SELOP  = 'addn'
ONLINE_TR = True
HSIZE    = 700
PRED_BATCH = 1
MTL_ETHA  = 1.0
BERT_OPTIM = 'adam' # ['adam' or 'rms']

SEQUENCE_LENGTH = 120

SIAM_BATCH   = 64
SIAM_SIZE    = 32
SIAM_DROPOUT = 0.0
SIAM_LR      = 5e-4 #2e-4
SIAM_EPOCH   = 50
K,M          = 3, 5 #2,3

def check_params(arg=None):
	global BATCH
	global LR1
	global LR2
	global EPOCHS
	global TEST_DATA_PATH
	global DATA_PATH
	global EVAL_DATA_PATH
	global ONLINE_TR
	global BERT_OPTIM
	global MTL_ETHA
	global HSIZE
	global SELOP
	global DPR

	INFOMAP_PATH = '/DATA/work_space/2-AI/3-SemEval21/infomap-master'
	INFOMAP_EX   = 'Infomap'
	OFFLINE_PATH = "/DATA/Mainstorage/Prog/NLP/vinai/bertweet-base"
	ONLINE_NAME  = "vinai/bertweet-base"

	parse = argparse.ArgumentParser(description='SemEval2021 Humor')
	parse.add_argument('-l1', dest='learning_rate_1', help='The firts learning rate to use in the optimizer', 
					   required=False, default=LR1)
	parse.add_argument('-l2', dest='learning_rate_2', help='The second learning rate to use in the optimizer', 
					   required=False, default=LR2)
	parse.add_argument('--dropout', dest='dropout', help='Dropout in the encoder', 
					   required=False, default=DPR)
	parse.add_argument('-s', dest='encod_size', help='The size of the dense layer in the encoder', 
					   required=False, default=HSIZE)
	parse.add_argument('-b', dest='batchs', help='Number of batchs', 
					   required=False, default=BATCH)
	parse.add_argument('-e', dest='epochs', help='Number of epochs', 
					   required=False, default=EPOCHS)
	parse.add_argument('-p', dest='predict', help='Unlabeled Data', 
					   required=False, default=TEST_DATA_PATH)
	parse.add_argument('-t', dest='train_data', help='Train Data', 
					   required=False, default=DATA_PATH)
	parse.add_argument('-d', dest='dev_data', help='Development Data', 
					   required=False, default=EVAL_DATA_PATH)
	parse.add_argument('--seed', dest='my_seed', help='Random Seed', 
					   required=False, default=12345)
	parse.add_argument('--infomap-path', dest='ipath', help='Path to infomap executable', 
					   required=False, default=INFOMAP_PATH)
	parse.add_argument('--infomap-name', dest='iname', help='Infomap executable name', 
					   required=False, default=INFOMAP_EX)
	parse.add_argument('--optim', dest='optim', help='Optimazer to use in train', 
					   required=False, default=BERT_OPTIM, choices=['adam', 'rms'])
	parse.add_argument('--vector-op', dest='selec', help='Operation to select the last vector from the transformet to fit the dense layer', 
					   required=False, default=SELOP, choices=['addn', 'first', 'mxp', 'att'])
	parse.add_argument('--etha', dest='etha', help='The multi task learning parameter to calculate the liner convex combination: \\math\\{loss = \\ethaL_1 + (1 - \\etha)L_2\\}', 
					   required=False, default=MTL_ETHA)		
	parse.add_argument('--offline', help='Use a local transformer, default False', 
					   required=False, action='store_false', default=True)
	parse.add_argument('--trans-offp', dest='offp', help='The local folder path to the pre-trained transformer', 
					   required=False, default=OFFLINE_PATH)
	parse.add_argument('--trans-onln', dest='onln', help='The transformers name', 
					   required=False, default=ONLINE_NAME)			   
   
	returns = parse.parse_args(arg)
	
	SELOP          = returns.selec 
	HSIZE          = int(returns.encod_size)
	MTL_ETHA       = float(returns.etha)
	LR1            = float(returns.learning_rate_1)
	LR2            = float(returns.learning_rate_2)
	BATCH          = int(returns.batchs)
	EPOCHS         = int(returns.epochs)
	TEST_DATA_PATH = returns.predict
	DATA_PATH      = returns.train_data
	EVAL_DATA_PATH = returns.dev_data
	ONLINE_TR      = bool(returns.offline)
	BERT_OPTIM     = returns.optim
	DPR            = float(returns.dropout)
	my_seed		   = int(returns.my_seed)
	
	# Set Infomap staf
	INFOMAP_EX = returns.iname 
	INFOMAP_PATH = returns.ipath
	setInfomapData(INFOMAP_PATH, INFOMAP_EX)

	# Set Transformers staf
	OFFLINE_PATH = returns.offp 
	ONLINE_NAME  = returns.onln
	setOfflinePath(OFFLINE_PATH)
	setOnlineName(ONLINE_NAME)

	if not os.path.isdir('data'):
		os.mkdir('data')
	if not os.path.isdir('pts'):
		os.mkdir('pts')
	if not os.path.isdir('pics'):
		os.mkdir('pics')
	if not os.path.isdir('preds'):
		os.mkdir('preds')
	
	torch.manual_seed(my_seed)
	np.random.seed(my_seed)
	random.seed(my_seed)
	
	if not ONLINE_TR:
		offline(True)

def clear_environment():
	delete_transformers()

def TrainRawEncoder():
	global DATA_PATH
	global EVAL_DATA_PATH
	global TEST_DATA_PATH

	t_data, t_loader = makeDataSet_Raw(DATA_PATH, batch=BATCH)
	e_data, e_loader = makeDataSet_Raw(EVAL_DATA_PATH, batch=BATCH)

	model = makeModels('bencoder', HSIZE, dpr=DPR, selection=SELOP)
	trainModels(model, t_loader, epochs=EPOCHS, evalData_loader=e_loader, etha=MTL_ETHA, mtl = False if MTL_ETHA >= 0.99 else True,
				nameu='roberta', optim=model.makeOptimizer(lr=LR1, lr_fin=LR2, algorithm=BERT_OPTIM))
	del t_loader
	del e_loader
	del t_data
	del e_data

	# Loading the best fit model
	model.load(os.path.join('pts', 'roberta.pt'))
	data, loader     = makeDataSet_Raw(TEST_DATA_PATH, batch=BATCH, shuffle=False)
	t_data, t_loader = makeDataSet_Raw(DATA_PATH, batch=BATCH, shuffle=False)
	e_data, e_loader = makeDataSet_Raw(EVAL_DATA_PATH, batch=BATCH, shuffle=False)

	# Make predictions using only the encoder
	evaluateModels(model, loader, name='pred_en')
	# Convert the data into vectors
	DATA_PATH      = convert2EncoderVec('train_en', model, t_loader, save_as_numpy=True)
	EVAL_DATA_PATH = convert2EncoderVec('dev_en', model, e_loader, save_as_numpy=True)
	TEST_DATA_PATH = convert2EncoderVec('test_en', model, loader, save_as_numpy=True)
	
	del t_loader
	del e_loader
	del t_data
	del e_data
	del loader
	del data 

def prep_Siam():
	DATA_PATH      = 'data/train_en.csv'
	EVAL_DATA_PATH = 'data/dev_en.csv'

	#C umbral=(0.00055, 0.0015), max_module=1)
	#B umbral=(0.00087, 0.00025), max_module=1)
	#A umbral=(0.0012, 0.0008), max_module=1)

	# findCenter_and_Limits(DATA_PATH, K,M, method='c-graph', method_distance='euclidea', umbral=(0.0013, 0.004), max_module=1)
	# findCenter_and_Limits(DATA_PATH, K,M, method='i-graph', method_distance='euclidea', umbral=(0.0012, 0.0008), max_module=1) #0.004
	# projectData2D(DATA_PATH, save_name='EncodC', use_centers=True) # 180 166

	# return 

	dts = makeSiamData(DATA_PATH, K, M, ref_folder='data', distance='euclidea')
	des = makeSiamData(EVAL_DATA_PATH, K, M, ref_folder='data', distance='euclidea')
	# dts = 'data/Siamtrain_en.csv'
	# des = 'data/Siamdev_en.csv'


	t_data, t_loader = makeDataSet_Siam(dts, batch=SIAM_BATCH)
	e_data, e_loader = makeDataSet_Siam(des, batch=SIAM_BATCH)

	model = makeModels('siam', SIAM_SIZE, dpr=SIAM_DROPOUT, in_size=64)
	trainModels(model, t_loader, epochs=SIAM_EPOCH, evalData_loader=e_loader,
				lr=SIAM_LR, nameu='siames', b_fun=min, smood=True, mtl=False, use_acc=False)

def pred_with_Siam():
	global TEST_DATA_PATH
	global EVAL_DATA_PATH
	global DATA_PATH
	global K
	global M 

	TEST_DATA_PATH = 'data/test_en.csv'
	EVAL_DATA_PATH = 'data/dev_en.csv'
	DATA_PATH = 'data/train_en.csv'

	model = makeModels('siam', SIAM_SIZE, dpr=SIAM_DROPOUT, in_size=64)
	model.load(os.path.join('pts', 'siames.pt'))
	
	DATA_PATH, K, M = makeSiam_ZData(DATA_PATH, model, ref_folder='data', batch=PRED_BATCH)
	EVAL_DATA_PATH, K, M = makeSiam_ZData(EVAL_DATA_PATH, model, ref_folder='data', batch=PRED_BATCH)
	TEST_DATA_PATH, K, M = makeSiam_ZData(TEST_DATA_PATH, model, ref_folder='data', batch=PRED_BATCH)
	
	predictManual(DATA_PATH, K, M, shost_compare=True)
	predictManual(EVAL_DATA_PATH, K, M, shost_compare=True)
	predictManual(TEST_DATA_PATH, K, M)

if __name__ == '__main__':
	check_params(arg=sys.argv[1:])

	TrainRawEncoder()
	# prep_Siam()
	# pred_with_Siam()

	#clear_environment()
