#%%
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append('../')
import os, pandas as pd, numpy as np
import tensorflow as tf, numpy as np
import keras as K
from utils import load_data_hateval, load_data_offenseval, maked_categorical_crossentropy, get_splits_for_val, f1
from utils import MaskedCategoricalAccuracy, load_data, masked_mse, root_mean_squared_error
from bert.tokenization import FullTokenizer
import tensorflow as tf
import keras as K
from utils_feature import Preprocess, convert_lines
from transformers import AutoTokenizer

def Minkowski_loss(y_true, y_pred):
    return K.backend.mean(K.backend.pow(K.backend.abs(y_pred-y_true), 1.2))


os.system('python generate_embedding.py')
maxlen = 70
np.random.seed(2021)
tf.random.set_seed(2021)

tokenizer = AutoTokenizer.from_pretrained("roberta-base") 

text, is_humor, humor_rating, controversy, offensiveness = load_data('../../data/train.csv')
text_dev, is_humor_dev, humor_rating_dev, controversy_dev, offensiveness_dev = load_data('../../data/dev.csv')


dic = np.load('affective_dic.npy', allow_pickle=True)
embedding_matrix = np.load('affective_embedding.npy')
indexes = x = dict(zip(dic, [i for i in range(len(dic))]))

text = convert_lines(text, 70, tokenizer, indexes)
text_dev = convert_lines(text_dev, 70, tokenizer, indexes)


def Recurrent_Feature(inshape, inshape1):

    I = K.layers.Input(inshape)
    emb = K.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                           embeddings_initializer=K.initializers.Constant(embedding_matrix), input_length=70, trainable=True, name='embd')(I)

    print('embedding shape', emb.shape)
    # emb = K.layers.Dropout(name='dp', rate=0.1)(emb1)
    F = K.layers.LSTM(units=emb.shape[2], name='foreward_lstm1', return_sequences=True)(emb)
    F  = F + emb
    F = K.layers.BatchNormalization(name='fnorm_lstm')(F)
    F = K.layers.LSTM(units=64, name='foreward_lstm2', return_sequences=True)(F)
    
    B = K.layers.LSTM(units=emb.shape[2], name='backwards_lstm1', go_backwards= True, return_sequences=True)(emb)
    B  = B + emb    
    B = K.layers.BatchNormalization(name='bnorm_lstm')(B)
    B = K.layers.LSTM(units=64, name='backwards_lstm2', go_backwards= True, return_sequences=True)(B)
    
    X = K.layers.concatenate([F, B], axis=2)
    print(X.shape)
    X = K.layers.Bidirectional(K.layers.LSTM(units=32, name='bilstm3'))(X)
    # X = K.layers.Dropout(name='dp', rate=0.1)(X)

    X = K.layers.Dense(64, activation='relu', name = 'extract_features')(X)
    X = K.layers.BatchNormalization(name= 'norm2' )(X)
    # X = K.layers.Dense(32, activation='relu', name = 'extract_features1')(X)
    # output_humor = K.layers.Dense(2, activation='softmax', name='output_humor')(X)
    output_funies = K.layers.Dense(1, activation='relu', name='output_funies')(X)
    # output_controversy = K.layers.Dense(2, activation='softmax', name='output_controversy')(X)
    output_offensiveness = K.layers.Dense(1, activation='relu', name='output_offensiveness')(X)

    return K.Model(inputs=[I], outputs=[  output_funies,   output_offensiveness])

losses = { 'output_funies':Minkowski_loss, 
            'output_offensiveness':Minkowski_loss}
metrics = { 'output_funies':root_mean_squared_error, 
            'output_offensiveness':root_mean_squared_error}

splits = 5
data_val = get_splits_for_val(is_humor, splits)
allindex = [i for i in range(len(offensiveness))]

encode_train = None 
encode_dev = None
encode_test = None

texttest = convert_lines(file[:, 1], 70, tokenizer, indexes)

for i in range(splits):
  # K.backend.clear_session()
  train_index = list(set(allindex) - set(data_val[i]))
  test_index = list(data_val[i])

  feature_extract = Recurrent_Feature(text[0].shape)
  # feature_extract.summary()
  filepath = "weightst.h5"
  checkpointer = K.callbacks.ModelCheckpoint(filepath, monitor='val_output_offensiveness_root_mean_squared_error', mode='min', verbose=1, save_best_only=True, save_weights_only =True)

  feature_extract.compile(optimizer=K.optimizers.Adam(lr=2e-2, decay=5e-3), loss=losses, metrics=metrics)
  history = feature_extract.fit([text] ,[ humor_rating, offensiveness],
            validation_data=([text_dev, text_dev_semantica],
                            [  humor_rating_dev,  offensiveness_dev]),
            batch_size=64, epochs=32, callbacks=[checkpointer], verbose=2)

  feature_extract.load_weights(filepath)
  feature_extract.evaluate([text_dev, text_dev_semantica],
                            [  humor_rating_dev,  offensiveness_dev], verbose=2)
  if i == 0:
    Embedder = K.models.Model(inputs=feature_extract.input, outputs=feature_extract.get_layer('extract_features').output)

    encode_dev = Embedder.predict([text_dev])
    encode_test = Embedder.predict([texttest])
    encode_train = Embedder.predict([text])
  else:
    Embedder = K.models.Model(inputs=feature_extract.input, outputs=feature_extract.get_layer('extract_features').output)

    encode_dev = (encode_dev + Embedder.predict([text_dev]))/2
    encode_test = (encode_test+ Embedder.predict([texttest]))/2
    encode_train = (encode_train+ Embedder.predict([text]))/2
  break

np.save('../../data/features/encode_public_dev', encode_dev)
np.save('../../data/features/encode_public_test', encode_test)
np.save('../../data/features/encode_public_train', encode_train)
