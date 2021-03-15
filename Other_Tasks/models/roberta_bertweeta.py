
import pandas as pd
import sys
import os
import keras as K
import tensorflow as tf
from transformers import AutoTokenizer, TFRobertaModel, TFAutoModel, TFAlbertModel, TFXLMModel
import numpy as np
from utils import f1, load_data, Evaluate, convert_lines, Save_Encode, convert_lines_for_BERT
from utils import load_data_sarcasm, get_splits_for_val, masked_mse, masked_root_mean_squared_error
from utils import masked_categortical_crossentropy, MaskedCategoricalAccuracy
from utils import trucncated_under_sampling, set_lt_multipliers
from Optimizers.RMS_Nest.NadamW import NadamW
from Optimizers.AdamLRM.adamlrm import AdamLRM
from keras_self_attention import SeqSelfAttention
from matplotlib import pyplot as plt


def SarcOffensive_build(shapei, shapef, model, maxlen):
  I = K.layers.Input(shapei, dtype='int32')
  features = K.layers.Input(shapef)
 
  hidden_states = model([I])[2][12]

  hidden_state = K.layers.Dropout(rate = 0.2, name='dp_robertas_outputs')(hidden_states)
  # print(hidden_states.shape)
 
  extractF = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,0])
  extractL = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,maxlen-1])
  # print(extractL.shape)
 
  add = tf.reduce_sum(hidden_states, axis=1, name = 'suma')
  norm = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(tf.math.l2_normalize(add))
 
  mxp = K.layers.MaxPooling1D(pool_size=maxlen,data_format='channels_last',name='maxpooling')(hidden_states)
  mxp = K.layers.Flatten(data_format="channels_first")(mxp)
  mxp = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(K.layers.BatchNormalization(name='batch_norm_concat')(mxp))

  extract = K.layers.concatenate([extractF, norm, mxp, extractL], axis = 2, name='Concatenate_BERT_output')
  extract = K.layers.Permute((2, 1))(extract)
  print(extract.shape)
  extract = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(extract)
  extract = tf.reduce_sum(extract, axis=1, name = 'suma')
  extract = K.layers.BatchNormalization(name='bout_batch_norm')(extract)
  print(extract.shape)
  # extract = scaled_dot_attention(name='att_feat')(cr)
  #probar con 32

  extract = K.layers.concatenate([extract, features], axis = 1, name='Concatenate_features')

  dense = K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                            ,name='encoder_layer')(extract)
  dense = K.layers.BatchNormalization(name='batch_norm')(dense)
                            
  # ff = K.layers.Input(fea)
  # features = K.layers.concatenate([dense, ff])

  output_humor = K.layers.Dense(2, activation='softmax', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_humor')(dense)
  output_funness = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_funness')(dense)
  # output_constroversy = K.layers.Dense(2, activation='softmax', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
  #                             , name='output_constroversy')(dense)
  # output_offensivenes = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
  #                             , name='output_offensivenes')(dense)
                              
  return K.Model(inputs=[I, features], outputs=[output_humor,output_funness])
 
np.random.seed(1)
tf.random.set_seed(1)

drive_path = '/content/drive/MyDrive/Semeval/Share'
local_path = '..'
path_prefix = local_path
maxlen = 70
DATA_PATH = path_prefix + '/data/train.csv'
 
tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
text, is_humor, humor_rating, controversy, offensiveness = load_data(DATA_PATH)
# text, is_humor, humor_rating, controversy, offensiveness = load_data(DATA_PATH)
text_dev, is_humor_dev, humor_rating_dev, controversy_dev, offensiveness_dev = load_data(path_prefix+'/dev.csv')


text_sarc, sarcasm = load_data_sarcasm(path_prefix+'/sarcasm2018_data.csv')
# ll1 = len(text_sarc)
textt = convert_lines(text, maxlen, tokenizer)  
text = convert_lines(text, maxlen, tokenizer)  
text_dev = convert_lines(text_dev, maxlen, tokenizer)  

def Minkowski_loss(y_true, y_pred):
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx())
  return K.backend.mean(K.backend.pow(K.backend.abs(y_pred-y_true), 1.5))#antes en 1.4, es lo mejor que teniams
 
def root_mean_squared_error(y_true, y_pred):
    return K.backend.sqrt(K.losses.mean_squared_error(y_true, y_pred))
        
def logcosh(y_true, y_pred):
    return K.backend.reduce_sum(K.backend.log(K.backend.cosh(y_true-y_pred)))
 
losses = { 'output_humor':K.losses.categorical_crossentropy, 'output_funness':Minkowski_loss}
metrics = {'output_humor': f1, 'output_funness':masked_root_mean_squared_error}
 
mean = 0
 

features_train = np.load(prefix_path + '/data/features/encode_train.csv.npy') 
featt = np.load(prefix_path + '/data/features/encode_train.csv.npy') 
features_dev =  np.load(prefix_path + '/data/features/encode_public_dev.csv.npy')
features_test =  np.load(prefix_path + '/data/features/encode_public_test.csv.npy')

features_eval = {'train.csv':features_train, 'public_test.csv': features_test, 'public_dev.csv': features_dev}


id_dev = pd.read_csv(path_prefix + '/public_dev.csv').to_numpy()[:,0]
file = pd.read_csv(path_prefix + '/public_test.csv').to_numpy() 
id_test = file[:,0]
text_test = file[:,1]
text_test = convert_lines(text_test, maxlen, tokenizer)

preds = {'humor':np.zeros((len(text_dev), )), 'rating':np.zeros((len(text_dev), )), 'cont':np.zeros((len(text_dev), )),'off':np.zeros((len(text_dev), )),}
preds_test = {'humor':np.zeros((len(text_test), )), 'rating':np.zeros((len(text_test), )), 'cont':np.zeros((len(text_test), )),'off':np.zeros((len(text_test), )),}

real_plot = []
pred_plot = []

encode_train = None 
encode_dev = None
encode_test = None 

model = TFRobertaModel.from_pretrained('/data/lm_fine_tunning',output_hidden_states=True)#TFRobertaModel.from_pretrained("roberta-base",output_hidden_states=True)
SarcOffensive = SarcOffensive_build(text[0].shape, features_train[0].shape,  model, maxlen)

coef_learning = set_lt_multipliers(0.9, SarcOffensive)    
opt = NadamW(learning_rate=1e-5, decay=2e-5, lr_multipliers=coef_learning, init_verbose=False)

SarcOffensive.compile(optimizer=opt, loss=losses, metrics=metrics)
filepath = "BERTWEET.h5"

checkpointer = K.callbacks.ModelCheckpoint(filepath, verbose=1, monitor='val_output_funness_masked_root_mean_squared_error', mode='min', save_best_only=True, save_weights_only =True)

history = SarcOffensive.fit([text, features_train], [ K.utils.to_categorical(is_humor), humor_rating],
            validation_data=([text_dev, features_dev], [K.utils.to_categorical(is_humor_dev), humor_rating_dev]),
            batch_size=32, epochs=15, callbacks=[checkpointer], verbose = 1)
  
SarcOffensive.load_weights(filepath, by_name=True)
###-------------- Average Predictions
# real_plot += list(offensiveness_dev)
# pred_plot += list(SarcOffensive.predict([text_dev, features_dev])[1])

predicts = SarcOffensive.predict([text_test, features_test])

preds_test['humor'] += np.argmax(predicts[0], axis = 1)
preds_test['rating'] += predicts[1].reshape(-1)
# preds_test['cont'] += np.argmax(predicts[2], axis = 1)
# preds_test['off'] += predicts[1].reshape(-1)


Embedder = K.models.Model(inputs=SarcOffensive.input, outputs=SarcOffensive.get_layer('encoder_layer').output)

encode_dev = Embedder.predict([text_dev, features_dev])
encode_test = Embedder.predict([text_test, features_test])
encode_train = Embedder.predict([textt, featt])

###-----------------Save Predictions
# preds_test['humor'] = np.array(np.round(preds_test['humor']), dtype=np.int)
# # preds_test['off'] = np.clip(preds_test['off'], 0, 5)
# preds_test['rating'] =  np.clip(preds_test['rating'], 0, 5)
# # preds['cont'] = np.array(np.round(preds_test['cont']/splits), dtype=np.int)

# # dictionary = {'id': id_test, 'is_humor': preds_test['humor'], 'humor_rating':preds_test['rating'], 'humor_controversy':preds_test['cont'], 'offense_rating':preds_test['off']}  

# dictionary = {'id': id_test, 'is_humor': preds_test['humor'], 'humor_rating':preds_test['rating']}  
# df = pd.DataFrame(dictionary) 

# df.to_csv('bertweet_off_test.csv')
# print('Evaluation Done!!') 
# ###-----------------Save Encodes

np.save(DATA_PATH[:-9] +'roberta_dev_encode', encode_dev)
np.save(DATA_PATH[:-9] +'roberta_test_encode', encode_test)
np.save(DATA_PATH[:-9] +'roberta_train_encode', encode_train)

# predicts = SarcOffensive.predict([text_dev, features_dev])[1].reshape(-1)

# dictionary = {'id': id_dev,  'humor_rating':np.clip(predicts, 0, 5)}  
# df = pd.DataFrame(dictionary) 

# df.to_csv('bertweet_off_dev.csv')
# print('Evaluation Dev Done!!')