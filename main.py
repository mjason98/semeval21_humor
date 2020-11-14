import os
from code.models import makeTrain_and_ValData, make_BertRep_from_data
from code.models import offline, load_transformers, delete_transformers
from code.models import makeModels, trainModels, makeDataSet_Vecs
from code.models import evaluateModels, makePredictData
from code.utils  import projectData2D


DATA_PATH       = 'data/train.csv'
EVAL_DATA_PATH  = ''
TEST_DATA_PATH  = 'data/dev.csv'

SEQUENCE_LENGTH = 120
BERT_BATCH      = 32
BATCH           = 64

ENCODER_BATCH   = 64
ENCODER_SIZE    = 800
ENCODER_DROPOUT = 0.2
ENCODER_LR      = 0.05
ENCODER_EPOCH   = 10

def prepare_environment():
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('pts'):
        os.mkdir('pts')
    if not os.path.isdir('pics'):
        os.mkdir('pics')
    
    offline(True) # cambiar esto
    #load_transformers()

def clear_environment():
    delete_transformers()

def TrainEncoder():
    t_data, t_loader = makeDataSet_Vecs(DATA_PATH, batch=ENCODER_BATCH)
    e_data, e_loader = makeDataSet_Vecs(EVAL_DATA_PATH, batch=ENCODER_BATCH)

    model = makeModels('encoder', ENCODER_SIZE, dpr=ENCODER_DROPOUT)
    trainModels(model, t_loader, epochs=ENCODER_EPOCH,evalData_loader=e_loader, lr=ENCODER_LR)

def Predict():
    data, loader = makePredictData(TEST_DATA_PATH, batch=BATCH)
    
    model = makeModels('encoder', ENCODER_SIZE, dpr=ENCODER_DROPOUT)
    model.load(os.path.join('pts', 'encoder.pt'))
    evaluateModels(model, loader)

if __name__ == '__main__':
    prepare_environment()

    # Spliting data
    DATA_PATH, EVAL_DATA_PATH = makeTrain_and_ValData(DATA_PATH, percent=10)
    
    # Making RoBERTa representations from data
    DATA_PATH      = make_BertRep_from_data(DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH)
    EVAL_DATA_PATH = make_BertRep_from_data(EVAL_DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH)
    TEST_DATA_PATH = make_BertRep_from_data(TEST_DATA_PATH, max_length=SEQUENCE_LENGTH, my_batch=BERT_BATCH,
                                            drops=[], final_drop=['text'], header_out=('id', 'x'))
    # Analisis
    #projectData2D(DATA_PATH, save_name='2Data')

    # Training the encoder
    #TrainEncoder()

    # Make predictions
    Predict()

    #clear_environment()