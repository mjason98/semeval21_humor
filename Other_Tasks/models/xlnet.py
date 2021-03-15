import sys
import os, pandas as pd
import keras as K
import tensorflow as tf
from transformers import AutoTokenizer, TFRobertaModel, TFXLNetModel
import numpy as np
from utils import f1, load_data, Evaluate, convert_lines, Save_Encodexln, load_data_sarcasm
from utils import load_data_offenseval, get_splits_for_val, masked_mse, masked_root_mean_squared_error
from utils import masked_categortical_crossentropy, MaskedCategoricalAccuracy
from utils import trucncated_under_sampling, set_lt_multipliers, set_lt_multipliers_xlnet
from keras_self_attention import SeqSelfAttention
from Optimizers.RMS_Nest.NadamW import NadamW
from Optimizers.AdamLRM.adamlrm import AdamLRM
from matplotlib import pyplot as plt
np.random.seed(1)
tf.random.set_seed(1)

drive_path = '/content/drive/MyDrive/Semeval/Share'
local_path = '..'
path_prefix = local_path
maxlen = 70
DATA_PATH = path_prefix + '/data/train.csv'

def TransOffensive_build(shapei, fi, model, maxlen):
  I = K.layers.Input(shapei, dtype='int32')

  hidden_states = model([I]).hidden_states[12]
  # hidden_state = K.layers.Dropout(rate = 0.1, name='dp_robertas_outputs')(hidden_states)
  # print(hidden_state.shape)

  extractF = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,0])
  extractL = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,maxlen-1])
  # # print(extractL.shape)

  add = tf.reduce_sum(hidden_states, axis=1, name = 'suma')
  
  norm = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(tf.math.l2_normalize(add))

  mxp = K.layers.MaxPooling1D(pool_size=maxlen,data_format='channels_last',name='maxpooling')(hidden_states)
  mxp = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(K.layers.Flatten(data_format="channels_first")(mxp))

  extract = K.layers.concatenate([extractF, norm, mxp, extractL], axis = 2, name='Concatenate__output')

  extract = K.layers.Permute((2, 1))(extract)
  print(extract.shape)
  extract = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(extract)
  extract = tf.reduce_sum(extract, axis=1, name = 'suma')
  extract = K.layers.BatchNormalization(name='bout_batch_norm')(extract)
  feat = K.layers.Input(fi)
  extract = K.layers.concatenate([extract, feat], axis = 1)
  print(extract.shape)

  dense = K.layers.Dense(64, activation='linear', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                            ,name='encoder_layer')(extract) 
  print(dense.shape)
  # output_sarcasm = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
  #                             , name='output_sarcasm')(dense)
  # output_funness = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
  #                             , name='output_funness')(dense)
  # output_constroversy = K.layers.Dense(2, activation='softmax', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
  #                             , name='output_constroversy')(dense)
  output_offensivenes = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_offensivenes')(dense)

  return K.Model(inputs=[I, feat], outputs=[ output_offensivenes])


tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased") 
text, is_humor, humor_rating, controversy, offensiveness = load_data(DATA_PATH)
text_dev, is_humor_dev, humor_rating_dev, controversy_dev, offensiveness_dev = load_data(path_prefix+'/dev.csv')

textt = convert_lines(text, maxlen, tokenizer)
text = convert_lines(text, maxlen, tokenizer)  
text_dev = convert_lines(text_dev, maxlen, tokenizer) 


print(sarcasm.shape, offensiveness.shape, text.shape)

def root_mean_squared_error(y_true, y_pred):
    return K.backend.sqrt(K.losses.mean_squared_error(y_true, y_pred))
		
def logcosh(y_true, y_pred):
	return K.backend.reduce_sum(K.backend.log(K.backend.cosh(y_true-y_pred)))

def Minkowski_loss(y_true, y_pred):
  return K.backend.mean(K.backend.pow(K.backend.abs(y_pred-y_true), 1.4))

def Minkowski_masked_loss(y_true, y_pred):
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx())
  return K.backend.mean(K.backend.pow(K.backend.abs(y_pred*mask-y_true*mask), 1.4))#antes en 1.4, es lo mejor que teniams
 
losses = { 'output_offensivenes':Minkowski_masked_loss}
metrics = { 'output_offensivenes': masked_root_mean_squared_error}

mean = 0

file = pd.read_csv(path_prefix + '/public_test.csv').to_numpy() 
id_test = file[:,0]
text_test = file[:,1]
text_test = convert_lines(text_test, maxlen, tokenizer)
id_dev = pd.read_csv(path_prefix + '/public_dev.csv').to_numpy()[:,0]

preds = {'humor':np.zeros((len(text_dev), )), 'rating':np.zeros((len(text_dev), )), 'cont':np.zeros((len(text_dev), )),'off':np.zeros((len(text_dev), )),}
preds_test = {'humor':np.zeros((len(text_test), )), 'rating':np.zeros((len(text_test), )), 'cont':np.zeros((len(text_test), )),'off':np.zeros((len(text_test), )),}

real_plot = []
pred_plot = []

encode_train = None 
encode_dev = None
encode_test = None
filepath = "XLNET.h5"


features_train = np.load(prefix_path + '/data/features/encode_train.csv.npy') 
featt = np.load(prefix_path + '/data/features/encode_train.csv.npy') 
features_dev =  np.load(prefix_path + '/data/features/encode_public_dev.csv.npy')
features_test =  np.load(prefix_path + '/data/features/encode_public_test.csv.npy')
features_eval = {'train.csv':features_train, 'public_test.csv': features_test, 'public_dev.csv': features_dev}

model = TFXLNetModel.from_pretrained("xlnet-base-cased",output_hidden_states=True,return_dict=True)
XLNOffensive = TransOffensive_build(text[0].shape, featt[0].shape, model, maxlen)
coef_learning = set_lt_multipliers_xlnet(0.9, XLNOffensive)

opt = NadamW(learning_rate=1e-5, decay=2e-6, lr_multipliers=coef_learning, init_verbose=0)

XLNOffensive.compile(optimizer=opt, loss=losses, metrics=metrics)
checkpointer = K.callbacks.ModelCheckpoint(filepath, verbose=1, monitor='val_masked_root_mean_squared_error', mode='min', save_best_only=True, save_weights_only =True)
#%%

history = XLNOffensive.fit([text, features_train],[    offensiveness],
            validation_data=([text_dev, features_dev],
                            [   offensiveness_dev]),
            batch_size=32, epochs=12, callbacks=[checkpointer], verbose=1)

###-------------- Average Predictions

XLNOffensive.load_weights(filepath)


real_plot += list(offensiveness_dev)
pred_plot += list(XLNOffensive.predict([text_dev, features_dev]))

predicts = XLNOffensive.predict([text_test, features_test])

# preds_test['humor'] += np.argmax(predicts[0], axis = 1)
# preds_test['rating'] += predicts[0].reshape(-1)
# preds_test['cont'] += np.argmax(predicts[2], axis = 1)
preds_test['off'] += predicts.reshape(-1)


Embedder = K.models.Model(inputs=XLNOffensive.input, outputs=XLNOffensive.get_layer('encoder_layer').output)

encode_dev = Embedder.predict([text_dev, features_dev])
encode_test = Embedder.predict([text_test, features_test])
encode_train = Embedder.predict([textt, featt])


###-----------------Save Predictions

# preds_test['humor'] = np.array(np.round(preds_test['humor']), dtype=np.int)
# preds_test['off'] = np.clip(preds_test['off'], 0, 5)
# # preds_test['rating'] =  np.clip(preds_test['rating'], 0, 5)
# # preds['cont'] = np.array(np.round(preds_test['cont']), dtype=np.int)

# # dictionary = {'id': id_test, 'is_humor': preds_test['humor'], 'humor_rating':preds_test['rating'], 'humor_controversy':preds_test['cont'], 'offense_rating':preds_test['off']}  

# dictionary = {'id': id_test, 'offense_rating':preds_test['off']}  
# df = pd.DataFrame(dictionary) 

# df.to_csv('xlnet_test.csv')
# print('Evaluation Done!!') 
###-----------------Save Encodes

np.save(DATA_PATH[:-9] +'xlnet_dev_encode', encode_dev)
np.save(DATA_PATH[:-9] + 'xlnet_test_encode', encode_test)
np.save(DATA_PATH[:-9] + 'xlnet_train_encode', encode_train)

predicts = XLNOffensive.predict([text_dev, features_dev]).reshape(-1)

dictionary = {'id': id_dev,  'offense_rating':np.clip(predicts, 0, 5)}  
df = pd.DataFrame(dictionary) 

df.to_csv('xlnet_dev.csv')
print('Evaluation Done!!')

# fig = plt.figure()
# colors = ['b', 'g', 'r', 'y', 'w']
# from utils import  Save_Encodexln
# plt.scatter(real_plot, pred_plot, c = 'b', label = 'predictions')
# plt.plot([i for i in range(6)], [i for i in range(6)], c = 'y', label = 'Pos')
# plt.xlabel('real values')
# plt.ylabel('predicted values')
# # np.save('/content/drive/MyDrive/Semeval/data/pred', preds)
# # # np.save('/content/drive/MyDrive/Semeval/data/real', real)
# # for files in ['train.csv', 'public_test.csv', 'public_dev.csv']:
# #   Save_Encodexln( TransOffensive, '/content/drive/MyDrive/Semeval/data/' + files, maxlen, tokenizer)

# plt.plot(history.history['masked_root_mean_squared_error']) 
# plt.plot(history.history['val_masked_root_mean_squared_error']) 
# plt.legend(['rmse_train', 'rmse_test'], loc='upper left')
# plt.ylabel('rmse')
# plt.xlabel('Epoch')
# plt.show()