import os
import torch
import numpy as np

from code.models import makeTrain_and_ValData, make_BertRep_from_data
from code.models import offline, load_transformers, delete_transformers
from code.models import makeModels, trainModels, makeDataSet_Vecs
from code.models import evaluateModels, makePredictData
from code.models import makeDataSet_Siam, trainSiamModel
from code.utils  import projectData2D
from code.siam   import findCenter_and_Limits, makeSiamData, convert2EncoderVec

# =============================================
DATA_PATH       = 'data/train.csv'
EVAL_DATA_PATH  = ''
TEST_DATA_PATH  = 'data/dev.csv'
#==============================================

SEQUENCE_LENGTH = 120
BERT_BATCH      = 32
BATCH           = 64

ENCODER_BATCH   = 64
ENCODER_SIZE    = 800
ENCODER_DROPOUT = 0.2
ENCODER_LR      = 0.05
ENCODER_EPOCH   = 20

SIAM_BATCH   = 64
SIAM_SIZE    = 256
SIAM_DROPOUT = 0.0
SIAM_LR      = 0.01
SIAM_EPOCH   = 10
K,M          = 1, 3

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
	
	offline(True) # cambiar esto

def clear_environment():
	delete_transformers()

def TrainEncoder():
	global DATA_PATH
	global EVAL_DATA_PATH
	global TEST_DATA_PATH

	t_data, t_loader = makeDataSet_Vecs(DATA_PATH, batch=ENCODER_BATCH)
	e_data, e_loader = makeDataSet_Vecs(EVAL_DATA_PATH, batch=ENCODER_BATCH)

	model = makeModels('encoder', ENCODER_SIZE, dpr=ENCODER_DROPOUT)
	trainModels(model, t_loader, epochs=ENCODER_EPOCH,evalData_loader=e_loader, lr=ENCODER_LR)

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
	global DATA_PATH
	global EVAL_DATA_PATH

	findCenter_and_Limits(DATA_PATH, K,M)
	DATA_PATH = makeSiamData(DATA_PATH, ref_folder='data')
	EVAL_DATA_PATH = makeSiamData(EVAL_DATA_PATH, ref_folder='data')

def TrainSiam():
	global DATA_PATH
	global DATA_PATH

	t_data, t_loader = makeDataSet_Siam(DATA_PATH, batch=SIAM_BATCH)
	e_data, e_loader = makeDataSet_Siam(EVAL_DATA_PATH, batch=SIAM_BATCH)

	model = makeModels('siam', SIAM_SIZE, dpr=SIAM_DROPOUT, in_size=ENCODER_SIZE//4)
	trainSiamModel(model, t_loader, epochs=SIAM_EPOCH,evalData_loader=e_loader, lr=SIAM_LR)

if __name__ == '__main__':
	prepare_environment()

	# Spliting data
	DATA_PATH, EVAL_DATA_PATH = makeTrain_and_ValData(DATA_PATH, percent=10)
	
	# Making RoBERTa representations from data
	load_transformers()
	DATA_PATH      = make_BertRep_from_data(DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH)
	EVAL_DATA_PATH = make_BertRep_from_data(EVAL_DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH)
	TEST_DATA_PATH = make_BertRep_from_data(TEST_DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH,
											drops=[], final_drop=['text'], header_out=('id', 'x'))
	delete_transformers()
	projectData2D(DATA_PATH, save_name='2Data')

	# Training the encoder and making reference vectors
	TrainEncoder()
	projectData2D(DATA_PATH, save_name='2Data_enc')
	prep_Siam()

	# DATA_PATH = 'data/Siamtrain_en.csv'
	# EVAL_DATA_PATH = 'data/Siameval_en.csv'

	TrainSiam()

	# Make predictions
	# Predict()

	#clear_environment()