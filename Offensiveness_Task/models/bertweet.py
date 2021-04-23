import pandas as pd, sys, os, numpy as np
import keras as K
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from utils import f1, load_data, Evaluate, convert_lines, Save_Encode, root_mean_squared_error
from utils import load_data_sarcasm, get_splits_for_val, masked_mse, masked_root_mean_squared_error
from utils import masked_categortical_crossentropy, MaskedCategoricalAccuracy
from utils import trucncated_under_sampling, set_lt_multipliers
from Optimizers.RMS_Nest.NadamW import NadamW
from Optimizers.AdamLRM.adamlrm import AdamLRM
from keras_self_attention import SeqSelfAttention
from matplotlib import pyplot as plt


def Bertweet_build_with_imf(shapei, model, maxlen):
  I = K.layers.Input(shapei, dtype='int32') 
 
  hidden_states = model([I])[2][12]

  hidden_states = K.layers.Dropout(rate = 0.2, name='dp_robertas_outputs')(hidden_states) 
 
  extractF = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,0])
  extractL = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,maxlen-1]) 
 
  add = tf.reduce_sum(hidden_states, axis=1, name = 'suma')
  norm = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(tf.math.l2_normalize(add))
 
  mxp = K.layers.MaxPooling1D(pool_size=maxlen,data_format='channels_last',name='maxpooling')(hidden_states)
  mxp = K.layers.Flatten(data_format="channels_first")(mxp)
  mxp = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(K.layers.BatchNormalization(name='batch_norm_concat')(mxp))

  extract = K.layers.concatenate([extractF, norm, mxp, extractL], axis = 2, name='Concatenate_BERT_output')
  extract = K.layers.Permute((2, 1))(extract)
 
  extract = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(extract)
  extract = tf.reduce_sum(extract, axis=1, name = 'suma')
  extract = K.layers.BatchNormalization(name='bout_batch_norm')(extract)
 
  
  dense = K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                            ,name='encoder_layer')(extract)
  dense = K.layers.BatchNormalization(name='batch_norm')(dense)
                            

  output_irony = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_irony')(dense)
  output_offensivenes = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_offensivenes')(dense)
                              
  return K.Model(inputs=[I], outputs=[output_irony, output_offensivenes])

def Minkowski_loss(y_true, y_pred):
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx())
  return K.backend.mean(K.backend.pow(K.backend.abs(y_pred-y_true), 1.5))#antes en 1.4, es lo mejor que teniams
          

np.random.seed(1)
tf.random.set_seed(1)

def train_bertweet(DATA_PATH_TRAIN, DATA_PATH_TEST, model_name, maxlen = 70):

  if model_name=='bt':
    pt_path = "vinai/bertweet-base"
  elif model_name=='rob':
    pt_path = "roberta-base"

  tokenizer = AutoTokenizer.from_pretrained(pt_path) 
  text, is_humor, _, _, offensiveness = load_data(DATA_PATH)
  text_sarc, sarcasm = load_data_sarcasm('../data/sarcasm2018_data.csv')
  
  ll1 = len(text_sarc)
  textt = convert_lines(text, maxlen, tokenizer)  
  text = convert_lines(text, maxlen, tokenizer)  

  splits = 5

  data_val = get_splits_for_val(is_humor, splits)
  allindex = [i for i in range(len(offensiveness))] 
  print(sarcasm.shape, offensiveness.shape, text.shape)

  
  mean = 0
  
  file = pd.read_csv(DATA_PATH_TRAIN)
  file = file.to_numpy()
  id_dev = file[:,0]
  text_dev = file[:,1]
  text_dev = convert_lines(text_dev, maxlen, tokenizer)

  file = pd.read_csv(DATA_PATH_TEST)
  file = file.to_numpy()
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

  for i in range(splits):
  
    train_index = np.array(list(set(allindex) - set(data_val[i])))
    test_index = np.array(list(data_val[i]))
    train_index = list(train_index[np.random.permutation(len(train_index))])
    test_index = list(test_index[np.random.permutation(len(test_index))])
  
    model = TFAutoModel.from_pretrained("vinai/bertweet-base",output_hidden_states=True)
    SarcOffensive = Bertweet_build_with_imf(text[0].shape, model, maxlen, features_train[0].shape)
    
    coef_learning = set_lt_multipliers(1, SarcOffensive)    
    opt = NadamW(learning_rate=1e-5, decay=2e-5, lr_multipliers=coef_learning, init_verbose=False)
  
    SarcOffensive.compile(optimizer=opt, loss=Minkowski_loss, metrics=masked_root_mean_squared_error)
    filepath = path_prefix + "/data/BERTWEET_humor.h5"
  
    checkpointer = K.callbacks.ModelCheckpoint(filepath, verbose=1, monitor='val_f1', mode='max', save_best_only=True, save_weights_only =True)
  
    history = SarcOffensive.fit([text[train_index,:], features_train[train_index,:]],
                      [  K.utils.to_categorical(is_humor[train_index])],
                validation_data=([text[test_index, :], features_train[test_index,:]],
                                [K.utils.to_categorical(is_humor[test_index])]),
                batch_size=32, epochs=12, callbacks=[checkpointer], verbose = 1)
    
    SarcOffensive.load_weights(filepath, by_name=True)
    ###-------------- Average Predictions
    real_plot += list(is_humor[test_index])
    pred_plot += list(SarcOffensive.predict([text[test_index, :], features_train[test_index,:]]))

    predicts = SarcOffensive.predict([text_dev, features_dev])

    preds['humor'] += np.argmax(predicts, axis = 1)
    # preds['rating'] += predicts[1].reshape(-1)
    # preds['cont'] += np.argmax(predicts[2], axis = 1)
    # preds['off'] += predicts[1].reshape(-1)

    predicts = SarcOffensive.predict([text_test, features_test])

    preds_test['humor'] += np.argmax(predicts, axis = 1)
    # preds_test['rating'] += predicts[1].reshape(-1)
    # preds_test['cont'] += np.argmax(predicts[2], axis = 1)
    # preds_test['off'] += predicts[1].reshape(-1)

    if i == 0:
      Embedder = K.models.Model(inputs=SarcOffensive.input, outputs=SarcOffensive.get_layer('encoder_layer').output)

      encode_dev = Embedder.predict([text_dev, features_dev])
      encode_test = Embedder.predict([text_test, features_test])
      encode_train = Embedder.predict([textt, featt])
    else:
      Embedder = K.models.Model(inputs=SarcOffensive.input, outputs=SarcOffensive.get_layer('encoder_layer').output)

      encode_dev = np.concatenate([encode_dev, Embedder.predict([text_dev, features_dev])], axis=1)
      encode_test = np.concatenate([encode_test, Embedder.predict([text_test, features_test])], axis=1)
      encode_train = np.concatenate([encode_train, Embedder.predict([textt, featt])], axis = 1)

  ###-----------------Save Predictions
  # preds['humor'] = np.array(np.round(preds['humor']/splits), dtype=np.int)
  # preds['off'] = np.clip(preds['off']/splits, 0, 5)
  # preds['rating'] =  np.clip(preds['rating']/splits, 0, 5)
  # preds['cont'] = np.array(np.round(preds['cont']/splits), dtype=np.int)

  # dictionary = {'id': id_dev, 'is_humor': preds['humor'], 'humor_rating':preds['rating'], 'humor_controversy':preds['cont'], 'offense_rating':preds['off']}  
  # dictionary = {'id': id_dev,'is_humor': preds['humor']}  
  # df = pd.DataFrame(dictionary) 

  # df.to_csv(path_prefix + '/data/bertweet_humor_dev.csv')
  # print('Evaluation Done!!') 

  # preds_test['humor'] = np.array(np.round(preds_test['humor']/splits), dtype=np.int)
  # # preds_test['off'] = np.clip(preds_test['off']/splits, 0, 5)
  # # preds_test['rating'] =  np.clip(preds_test['rating']/splits, 0, 5)
  # # preds['cont'] = np.array(np.round(preds_test['cont']/splits), dtype=np.int)

  # # dictionary = {'id': id_test, 'is_humor': preds_test['humor'], 'humor_rating':preds_test['rating'], 'humor_controversy':preds_test['cont'], 'offense_rating':preds_test['off']}  

  # dictionary = {'id': id_test,  'is_humor': preds_test['humor']}  
  # df = pd.DataFrame(dictionary) 

  # df.to_csv(path_prefix + '/data/bertweet_humor_test.csv')
  # print('Evaluation Done!!') 
  ###-----------------Save Encodes

  np.save(DATA_PATH[:-9] + 'bertweet_humor_dev_encode', encode_dev)
  np.save(DATA_PATH[:-9] + 'bertweet_humor_test_encode', encode_test)
  np.save(DATA_PATH[:-9] + 'bertweet_humor_train_encode', encode_train)

  # plt.plot(history.history['output_offensivenes_root_mean_squared_error'])
  # plt.plot(history.history['val_output_offensivenes_root_mean_squared_error'])
  # plt.ylabel('mse')
  # plt.xlabel('Epoch')
  # plt.legend(['output_offensivenes_loss', 'val_output_offensivenes_loss'], loc='upper left')vinai/bertweetvinai/bertweet-base-base
  # plt.show()

  # matrix = sklearn.metrics.confusion_matrix(real_plot, pred_plot)
  # ax= plt.subplot()
  # seaborn.heatmap(matrix, cmap="YlGnBu",annot=True, ax = ax, square=False,robust=True, fmt='.4g')

  # # # labels, title and ticks
  # ax.set_xlabel('Predicted labels')
  # ax.set_ylabel('True labels')
  # ax.set_title('Confusion Matrix')
  # plt.show()

