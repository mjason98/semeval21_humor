import os
import sys
import torch
import argparse
import numpy as np

from code.models import makeTrain_and_ValData
from code.models import offline, load_transformers, delete_transformers
from code.models import makeModels, trainModels, makeDataSet_Vecs
from code.models import evaluateModels, makePredictData
from code.models import makeDataSet_Siam, makeDataSet_Raw
from code.utils  import projectData2D
from code.siam   import makeSiam_ZData, predictManual, setInfomapData
from code.siam   import findCenter_and_Limits, makeSiamData, convert2EncoderVec

# =============================================
DATA_PATH       = 'data/train.csv'
EVAL_DATA_PATH  = ''
TEST_DATA_PATH  = 'data/public_dev.csv'
#==============================================

BATCH  = 64
EPOCHS = 12
LR 	   = 3e-5
ONLINE_TR = True
HSIZE    = 800
PRED_BATCH = 16

SEQUENCE_LENGTH = 120

SIAM_BATCH   = 64
SIAM_SIZE    = 32
SIAM_DROPOUT = 0.0
SIAM_LR      = 0.001
SIAM_EPOCH   = 200
K,M          = 3, 5

def check_params(arg=None):
	global BATCH
	global LR
	global EPOCHS
	global TEST_DATA_PATH
	global DATA_PATH
	global EVAL_DATA_PATH
	global ONLINE_TR

	INFOMAP_PATH = '/DATA/work_space/2-AI/3-SemEval21/infomap-master'
	INFOMAP_EX   = 'Infomap'

	parse = argparse.ArgumentParser(description='SemEval2021 Humor')
	parse.add_argument('-l', dest='learning_rate', help='The learning rate to use in the optimizer', 
					   required=False, default=LR)
	parse.add_argument('-b', dest='batchs', help='Amount of batchs', 
					   required=False, default=BATCH)
	parse.add_argument('-e', dest='epochs', help='Amount of epochs', 
					   required=False, default=EPOCHS)
	parse.add_argument('-p', dest='predict', help='Unlabeled Data', 
					   required=False, default=TEST_DATA_PATH)
	parse.add_argument('-t', dest='train_data', help='Train Data', 
					   required=False, default=DATA_PATH)
	parse.add_argument('-d', dest='dev_data', help='Development Data', 
					   required=False, default=EVAL_DATA_PATH)
	parse.add_argument('--infomap-path', dest='ipath', help='Path to infomap executable', 
					   required=False, default=INFOMAP_PATH)
	parse.add_argument('--infomap-name', dest='iname', help='Infomap executable name', 
					   required=False, default=INFOMAP_EX)
	parse.add_argument('--offline', help='Use a local transformer, default False', 
					   required=False, action='store_false', default=True)
   
	returns = parse.parse_args(arg)

	LR    = float(returns.learning_rate)
	BATCH = int(returns.batchs)
	EPOCHS = int(returns.epochs)
	TEST_DATA_PATH = returns.predict
	DATA_PATH = returns.train_data
	EVAL_DATA_PATH = returns.dev_data
	ONLINE_TR = bool(returns.offline)
	
	# Set Infomap staf
	setInfomapData(INFOMAP_PATH, INFOMAP_EX)
	INFOMAP_EX = returns.iname 
	INFOMAP_PATH = returns.ipath

	if not os.path.isdir('data'):
		os.mkdir('data')
	if not os.path.isdir('pts'):
		os.mkdir('pts')
	if not os.path.isdir('pics'):
		os.mkdir('pics')
	if not os.path.isdir('preds'):
		os.mkdir('preds')
	
	torch.manual_seed(12345)
	np.random.seed(12345)
	
	if not ONLINE_TR:
		offline(True)

def clear_environment():
	delete_transformers()

def makeFinalData_Model():
	global DATA_PATH
	global EVAL_DATA_PATH
	global TEST_DATA_PATH

	model = makeModels('siam', SIAM_SIZE, dpr=SIAM_DROPOUT, in_size=ENCODER_SIZE//4)
	model.load(os.path.join('pts', 'siam.pt'))

	DATA_PATH = makeSiam_ZData(DATA_PATH, model, ref_folder='data', batch=CZ_BATCH)
	EVAL_DATA_PATH = makeSiam_ZData(EVAL_DATA_PATH, model, ref_folder='data', batch=CZ_BATCH)
	TEST_DATA_PATH = makeSiam_ZData(TEST_DATA_PATH, model, ref_folder='data', batch=CZ_BATCH)

	t_data, t_loader = makeDataSet_Vecs(DATA_PATH, batch=Z_BATCH)
	e_data, e_loader = makeDataSet_Vecs(EVAL_DATA_PATH, batch=Z_BATCH)
	
	model = makeModels('zmod', 1, dpr=0., in_size=int(K*2)) #in_size=int((K+M)*2))
	trainModels(model, t_loader, epochs=Z_EPOCH,evalData_loader=e_loader, 
				lr=Z_LR, nameu='zmod')

def TrainRawEncoder():
	global DATA_PATH
	global EVAL_DATA_PATH
	global TEST_DATA_PATH

	t_data, t_loader = makeDataSet_Raw(DATA_PATH, batch=BATCH)
	e_data, e_loader = makeDataSet_Raw(EVAL_DATA_PATH, batch=BATCH)

	model = makeModels('bencoder', HSIZE, dpr=0.0)
	trainModels(model, t_loader, epochs=EPOCHS, evalData_loader=e_loader,
				nameu='roberta', optim=model.makeOptimizer(lr=LR))

	del t_loader
	del e_loader
	del t_data
	del e_data

	# Loading the best fit model
	model.load(os.path.join('pts', 'roberta.pt'))
	data, loader     = makeDataSet_Raw(TEST_DATA_PATH, batch=BATCH, shuffle=False)
	t_data, t_loader = makeDataSet_Raw(DATA_PATH, batch=BATCH, shuffle=False)
	e_data, e_loader = makeDataSet_Raw(EVAL_DATA_PATH, batch=BATCH, shuffle=False)

	evaluateModels(model, loader, cleaner=['humor_rating'], name='pred_en')
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
	# DATA_PATH      = 'data/train_en.csv'
	# EVAL_DATA_PATH = 'data/dev_en.csv'

	findCenter_and_Limits(DATA_PATH, K,M, method='i-graph', method_distance='euclidea', umbral=0.1)
	projectData2D(DATA_PATH, save_name='2Data_igraph', use_centers=True)

	dts = makeSiamData(DATA_PATH, K, M, ref_folder='data', distance='euclidea')
	des = makeSiamData(EVAL_DATA_PATH, K, M, ref_folder='data', distance='euclidea')
	# dts = 'data/Siamtrain_en.csv'
	# des = 'data/Siamdev_en.csv'
	
	t_data, t_loader = makeDataSet_Siam(dts, batch=SIAM_BATCH)
	e_data, e_loader = makeDataSet_Siam(des, batch=SIAM_BATCH)

	model = makeModels('siam', SIAM_SIZE, dpr=SIAM_DROPOUT, in_size=64)
	trainModels(model, t_loader, epochs=SIAM_EPOCH, evalData_loader=e_loader,
				lr=SIAM_LR, nameu='siames', b_fun=min, smood=True)

def pred_with_Siam():
	global TEST_DATA_PATH
	global EVAL_DATA_PATH
	global DATA_PATH
	global K
	global M 

	# TEST_DATA_PATH = 'data/test_en.csv'
	# EVAL_DATA_PATH = 'data/dev_en.csv'
	# DATA_PATH = 'data/train_en.csv'

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

	# Spliting data
	DATA_PATH, EVAL_DATA_PATH = makeTrain_and_ValData(DATA_PATH, percent=10)

	TrainRawEncoder()
	prep_Siam()
	pred_with_Siam()
	
	# makeFinalData_Model()

	#clear_environment()