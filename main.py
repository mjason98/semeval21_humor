import os
import torch
import numpy as np

from code.models import makeTrain_and_ValData, make_BertRep_from_data
from code.models import offline, load_transformers, delete_transformers
from code.models import makeModels, trainModels, makeDataSet_Vecs
from code.models import evaluateModels, makePredictData
from code.models import makeDataSet_Siam, trainSiamModel, makeDataSet_Raw
from code.utils  import projectData2D
from code.siam   import makeSiam_ZData
from code.siam   import findCenter_and_Limits, makeSiamData, convert2EncoderVec

# =============================================
DATA_PATH       = 'data/train.csv'
EVAL_DATA_PATH  = ''
TEST_DATA_PATH  = 'data/public_dev.csv'
#==============================================

BATCH  = 2
EPOCHS = 1
LR 	   = 1e-5

SEQUENCE_LENGTH = 120
BERT_BATCH      = 32
CZ_BATCH        = 16

Z_BATCH 	    = 64 #128
Z_LR   			= 0.1
Z_EPOCH			= 200

ENCODER_BATCH   = 64
ENCODER_SIZE    = 800
ENCODER_DROPOUT = 0.2
ENCODER_LR      = 0.05
ENCODER_EPOCH   = 20

SIAM_BATCH   = 64
SIAM_SIZE    = 256
SIAM_DROPOUT = 0.0
SIAM_LR      = 0.001
SIAM_EPOCH   = 10
K,M          = 12, 5

def prepare_environment():
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
	
	# offline(True) # cambiar esto

def clear_environment():
	delete_transformers()

def TrainEncoder():
	global DATA_PATH
	global EVAL_DATA_PATH
	global TEST_DATA_PATH

	t_data, t_loader = makeDataSet_Vecs(DATA_PATH, batch=ENCODER_BATCH)
	e_data, e_loader = makeDataSet_Vecs(EVAL_DATA_PATH, batch=ENCODER_BATCH)

	model = makeModels('encoder', ENCODER_SIZE, dpr=ENCODER_DROPOUT)
	# trainModels(model, t_loader, epochs=ENCODER_EPOCH,evalData_loader=e_loader, lr=ENCODER_LR)

	# Convert Bert vectors to encoder vectors
	model.load(os.path.join('pts', 'encoder.pt'))
	data, loader = makePredictData(TEST_DATA_PATH, batch=ENCODER_BATCH)

	# Cuando libereb el dev con label, mirar los nombres abajo
	DATA_PATH      = convert2EncoderVec('train_en', model, t_loader)
	EVAL_DATA_PATH = convert2EncoderVec('eval_en', model, e_loader)
	TEST_DATA_PATH = convert2EncoderVec('dev_en', model, loader)

	del model
	del loader 
	del data 
	del t_loader
	del t_data
	del e_loader
	del e_data

def Predict():
	data, loader = makePredictData(TEST_DATA_PATH, batch=BATCH)
	# recordar hacer una funcion especial para preciccion
	# TEST_DATA_PATH = makeSiamData(TEST_DATA_PATH, ref_folder='data')
	
	model = makeModels('encoder', ENCODER_SIZE, dpr=ENCODER_DROPOUT)
	model.load(os.path.join('pts', 'encoder.pt'))
	evaluateModels(model, loader)

def prep_Siam():
	findCenter_and_Limits(DATA_PATH, K,M)
	dts = makeSiamData(DATA_PATH, ref_folder='data')
	des = makeSiamData(EVAL_DATA_PATH, ref_folder='data')
	return dts, des

def TrainSiam(dts, des):
	t_data, t_loader = makeDataSet_Siam(dts, batch=SIAM_BATCH)
	e_data, e_loader = makeDataSet_Siam(des, batch=SIAM_BATCH)

	model = makeModels('siam', SIAM_SIZE, dpr=SIAM_DROPOUT, in_size=ENCODER_SIZE//4)
	trainSiamModel(model, t_loader, epochs=SIAM_EPOCH,evalData_loader=e_loader, lr=SIAM_LR)

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
	t_data, t_loader = makeDataSet_Raw(DATA_PATH, batch=BATCH)
	e_data, e_loader = makeDataSet_Raw(EVAL_DATA_PATH, batch=BATCH)

	model = makeModels('bencoder', 800, dpr=0.0)
	model.save(os.path.join('pts', 'roberta.pt'))
	# trainModels(model, t_loader, epochs=EPOCHS, evalData_loader=e_loader,
	# 			nameu='roberta', optim=model.makeOptimizer(lr=LR))

	del t_loader
	del e_loader
	del t_data
	del e_data

	data , loader = makeDataSet_Raw(TEST_DATA_PATH, batch=BATCH, shuffle=False)
	model.load(os.path.join('pts', 'roberta.pt'))
	evaluateModels(model, loader, cleaner=['humor_rating'])

if __name__ == '__main__':
	prepare_environment()

	# Spliting data
	DATA_PATH, EVAL_DATA_PATH = makeTrain_and_ValData(DATA_PATH, percent=10)

	TrainRawEncoder()
	
	# Making RoBERTa representations from data
	# load_transformers()
	# DATA_PATH      = make_BertRep_from_data(DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH)
	# EVAL_DATA_PATH = make_BertRep_from_data(EVAL_DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH)
	# TEST_DATA_PATH = make_BertRep_from_data(TEST_DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH,
	# 										drops=[], final_drop=['text'], header_out=('id', 'x'))
	# delete_transformers()
	# projectData2D(DATA_PATH, save_name='2Data')

	# Training the encoder and making reference vectors
	# TrainEncoder()
	# projectData2D(DATA_PATH, save_name='2Data_enc')
	# dts, des = prep_Siam()
	# TrainSiam(dts, des)
	# makeFinalData_Model()

	# Make predictions
	# Predict()

	#clear_environment()