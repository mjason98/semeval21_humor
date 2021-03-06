import sys
import os, pandas as pd
import keras as K
import tensorflow as tf
from transformers import AutoTokenizer, TFXLNetModel
import numpy as np
from utils import get_splits_for_val, load_data, convert_lines, Minkowski_masked_loss, load_data_sarcasm
from utils import set_lt_multipliers_xlnet, masked_root_mean_squared_error
from keras_self_attention import SeqSelfAttention
from Optimizers.RMS_Nest.NadamW import NadamW
from matplotlib import pyplot as plt
np.random.seed(1)
tf.random.set_seed(1)

def TransOffensive_build(shapei, fi, model, maxlen):
  I = K.layers.Input(shapei, dtype='int32')

  hidden_states = model([I]).hidden_states[12]

  extractF = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,0])
  extractL = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(hidden_states[:,maxlen-1])

  add = tf.reduce_sum(hidden_states, axis=1, name = 'suma')
  
  norm = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(tf.math.l2_normalize(add))

  mxp = K.layers.MaxPooling1D(pool_size=maxlen,data_format='channels_last',name='maxpooling')(hidden_states)
  mxp = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(K.layers.Flatten(data_format="channels_first")(mxp))

  extract = K.layers.concatenate([extractF, norm, mxp, extractL], axis = 2, name='Concatenate__output')

  extract = K.layers.Permute((2, 1))(extract)
 
  extract = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(extract)
  extract = tf.reduce_sum(extract, axis=1, name = 'suma')
  extract = K.layers.BatchNormalization(name='bout_batch_norm')(extract)
  feat = K.layers.Input(fi)
  extract = K.layers.concatenate([extract, feat], axis = 1)
 
  extract = K.layers.Dense(64, activation='linear', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                            ,name='encoder_layer')(extract)

  output_irony = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_irony')(extract)
  output_offensivenes = K.layers.Dense(1, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                              , name='output_offensivenes')(extract)
                              
  return K.Model(inputs=[I], outputs=[output_irony, output_offensivenes])

def xlnet_process(DATA_PATH_TRAIN=None, DATA_PATH_TEST=None, phase='train', splits=5, maxlen = 70):

  tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased") 

  if phase in ['train', 'both']:
    text, is_humor, _, _, offensiveness = load_data(DATA_PATH_TRAIN)
    text_sarc, irony = load_data_sarcasm('../data/sarcasm2018_data.csv')
    
    textt = convert_lines(text, maxlen, tokenizer)  
    text = convert_lines(text, maxlen, tokenizer)  

    text = np.concatenate([textt, text_sarc])
    offensiveness = np.concatenate([offensiveness, np.zeros_like(irony)-1])
    irony = np.concatenate([np.zeros_like(is_humor)-1, np.zeros_like(irony)-1])
    perm = np.random.permutation(len(text))
    text = text[perm], offensiveness = offensiveness[perm], irony = irony[perm]
    print(irony.shape, offensiveness.shape, text.shape)

    data_val = get_splits_for_val(is_humor, splits)
    allindex = [i for i in range(len(offensiveness))] 
  
  if phase in ['encode', 'both']:
    text_dev = pd.read_csv(DATA_PATH_TEST).to_numpy()[:,1]
    text_dev = convert_lines(text_dev, maxlen, tokenizer)

  encode_train = None 
  encode_dev = None

  if phase in ['train', 'both']:
    for i in range(splits):
    
      train_index = np.array(list(set(allindex) - set(data_val[i])))
      test_index = np.array(list(data_val[i]))
      train_index = list(train_index[np.random.permutation(len(train_index))])
      test_index = list(test_index[np.random.permutation(len(test_index))])
    
      model = TFXLNetModel.from_pretrained("xlnet-base-cased",output_hidden_states=True,return_dict=True)
      XLNOffensive = TransOffensive_build(text[0].shape, featt[0].shape, model, maxlen)
      coef_learning = set_lt_multipliers_xlnet(0.9, XLNOffensive)

      opt = NadamW(learning_rate=1e-5, decay=2e-6, lr_multipliers=coef_learning, init_verbose=0)

      XLNOffensive.compile(optimizer=opt, loss=Minkowski_masked_loss, metrics=masked_root_mean_squared_error)
      filepath = f'../data/xlnet_weights{i+1}.h5'
    
      checkpointer = K.callbacks.ModelCheckpoint(filepath, verbose=1, monitor='val_f1', mode='max', save_best_only=True, save_weights_only =True)
    
      XLNOffensive.fit(text[train_index], [irony[train_index], offensiveness[train_index]],
                  validation_data=(text[test_index], [irony[test_index], offensiveness[test_index]]),
                  batch_size=32, epochs=12, callbacks=[checkpointer], verbose = 1)
  
  if phase in ['encode', 'both']:

    model = TFXLNetModel.from_pretrained("xlnet-base-cased",output_hidden_states=True,return_dict=True)
    XLNOffensive = TransOffensive_build(text[0].shape, featt[0].shape, model, maxlen)
      
    for i in range(splits):
      XLNOffensive.load_weights(f'../data/xlnet_weights{i+1}.h5', by_name=True)

      if i == 0:
        Embedder = K.models.Model(inputs=XLNOffensive.input, outputs=XLNOffensive.get_layer('encoder_layer').output)
  
        encode_dev = Embedder.predict(text_dev)
        if phase == 'both':
          encode_train = Embedder.predict(textt)
      else:
        Embedder = K.models.Model(inputs=XLNOffensive.input, outputs=XLNOffensive.get_layer('encoder_layer').output)

        encode_dev = np.concatenate([encode_dev, Embedder.predict([text_dev])], axis=1)
        if phase == 'both':
          encode_train = np.concatenate([encode_train, Embedder.predict([textt])], axis = 1)

  if encode_train is not None :
    np.save(f'../data/xlnet_train_encode', encode_dev)
  if encode_dev is not None:
    np.save(f'../data/xlnet_dev_encode', encode_train)