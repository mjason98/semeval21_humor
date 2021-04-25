import pandas as pd, sys, os, numpy as np
import keras as K
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from utils import f1, load_data, Evaluate, convert_lines, Minkowski_masked_loss
from utils import load_data_sarcasm, get_splits_for_val,  masked_root_mean_squared_error
from utils import trucncated_under_sampling, set_lt_multipliers
from Optimizers.RMS_Nest.NadamW import NadamW
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

np.random.seed(1)
tf.random.set_seed(1)

def bert_based(model_name, DATA_PATH_TRAIN=None, DATA_PATH_TEST=None, phase='train', splits=5, maxlen = 70):

  if model_name=='bt':
    pt_path = "vinai/bertweet-base"
  elif model_name=='rob':
    pt_path = "roberta-base"

  tokenizer = AutoTokenizer.from_pretrained(pt_path) 

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
    
      model = TFAutoModel.from_pretrained(f'../data/lm_fine_tunning_{model_name}',output_hidden_states=True)
      SarcOffensive = Bertweet_build_with_imf(text[0].shape, model, maxlen)
      
      coef_learning = set_lt_multipliers(1, SarcOffensive)    
      opt = NadamW(learning_rate=1e-5, decay=2e-5, lr_multipliers=coef_learning, init_verbose=False)
    
      SarcOffensive.compile(optimizer=opt, loss=Minkowski_masked_loss, metrics=masked_root_mean_squared_error)
      filepath = f'../data/{model_name}_weights{i+1}.h5'
    
      checkpointer = K.callbacks.ModelCheckpoint(filepath, verbose=1, monitor='val_f1', mode='max', save_best_only=True, save_weights_only =True)
    
      SarcOffensive.fit(text[train_index,:], [irony[train_index,:], offensiveness[train_index,:]],
                  validation_data=(text[test_index, :], [irony[test_index,:], offensiveness[test_index,:]]),
                  batch_size=32, epochs=12, callbacks=[checkpointer], verbose = 1)
  
  if phase in ['encode', 'both']:

    model = TFAutoModel.from_pretrained(f'../data/lm_fine_tunning_{model_name}',output_hidden_states=True)
    SarcOffensive = Bertweet_build_with_imf(text[0].shape, model, maxlen)
    
    for i in range(splits):
      SarcOffensive.load_weights(f'../data/{model_name}_weights{i+1}.h5', by_name=True)

      if i == 0:
        Embedder = K.models.Model(inputs=SarcOffensive.input, outputs=SarcOffensive.get_layer('encoder_layer').output)
  
        encode_dev = Embedder.predict(text_dev)
        if phase == 'both':
          encode_train = Embedder.predict(textt)
      else:
        Embedder = K.models.Model(inputs=SarcOffensive.input, outputs=SarcOffensive.get_layer('encoder_layer').output)

        encode_dev = np.concatenate([encode_dev, Embedder.predict([text_dev])], axis=1)
        if phase == 'both':
          encode_train = np.concatenate([encode_train, Embedder.predict([textt])], axis = 1)

  if encode_train is not None :
    np.save(f'../data/{model_name}_train_encode', encode_dev)
  if encode_dev is not None:
    np.save(f'../data/{model_name}_dev_encode', encode_train)
