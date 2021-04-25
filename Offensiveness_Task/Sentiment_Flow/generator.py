import sys, os, pandas as pd, numpy as np
sys.path.append('../models')
import tensorflow as tf, keras as K
from utils import  get_splits_for_val, Minkowski_loss
from utils import load_data, root_mean_squared_error
from transformers import AutoTokenizer

def convert_lines(example, max_seq_length ,tokenizer, indexes):
    
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
        # print(example[i])
        tokens_a = tokenizer.tokenize(example[i])

        if len(tokens_a ) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1

        one_token = []
        for j in tokens_a:
            if indexes.get(j) is not None:
                one_token.append(indexes[j])
            else: one_token.append(len(indexes))

        one_token = one_token +[len(indexes)]*(max_seq_length - len(tokens_a))

        all_tokens.append(one_token)
        
    print(longer)
    return np.array(all_tokens, dtype = np.int32)


def Recurrent_Feature(inshape, embedding_matrix):

    I = K.layers.Input(inshape)
    emb = K.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                           embeddings_initializer=K.initializers.Constant(embedding_matrix), input_length=70, trainable=True, name='embd')(I)

    print('embedding shape', emb.shape)
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

    X = K.layers.Dense(64, activation='relu', name = 'extract_features')(X)
    X = K.layers.BatchNormalization(name= 'norm2' )(X)
    output_funies = K.layers.Dense(1, activation='relu', name='output_funies')(X)
    output_offensiveness = K.layers.Dense(1, activation='relu', name='output_offensiveness')(X)

    return K.Model(inputs=[I], outputs=[  output_funies,   output_offensiveness])

def generate_sentiment_flow(DATA_PATH_TRAIN=None, DATA_PATH_TEST=None, phase='train', maxlen = 70):

    os.system('python generate_embedding.py')
    tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
    dic = np.load('affective_dic.npy', allow_pickle=True)
    embedding_matrix = np.load('affective_embedding.npy')
    indexes = x = dict(zip(dic, [i for i in range(len(dic))]))

    if phase in ['train', 'both']:
        text, is_humor, humor_rating, _, offensiveness = load_data(DATA_PATH_TRAIN)
        text = convert_lines(text, 70, tokenizer, indexes)
    
    if phase in ['encode', 'both']:
        text_dev, _, humor_rating_dev, _, offensiveness_dev = load_data(DATA_PATH_TEST)
        text_dev = convert_lines(text_dev, 70, tokenizer, indexes)
    
    if phase in ['train', 'both']:
        losses = { 'output_funies':Minkowski_loss, 'output_offensiveness':Minkowski_loss}
        metrics = { 'output_funies':root_mean_squared_error, 'output_offensiveness':root_mean_squared_error}

        data_val = get_splits_for_val(is_humor, 5)
        allindex = [i for i in range(len(offensiveness))]

        encode_train = None 
        encode_dev = None


        train_index = list(set(allindex) - set(data_val[i]))
        test_index = list(data_val[i])

        feature_extract = Recurrent_Feature(text[0].shape, embedding_matrix)

        filepath = "weightst.h5"
        checkpointer = K.callbacks.ModelCheckpoint(filepath, monitor='val_output_offensiveness_root_mean_squared_error', mode='min', verbose=1, save_best_only=True, save_weights_only =True)
        feature_extract.compile(optimizer=K.optimizers.Adam(lr=2e-2, decay=5e-3), loss=losses, metrics=metrics)
        
        feature_extract.fit(text[train_index],[ humor_rating[train_index], offensiveness[train_index]],
                    validation_data=(text[test_index],[ humor_rating[test_index], offensiveness[test_index]]),
                    batch_size=64, epochs=32, callbacks=[checkpointer], verbose=2)

    if phase in ['encode', 'both']:
 
        feature_extract.load_weights("weightst.h5")
        Embedder = K.models.Model(inputs=feature_extract.input, outputs=feature_extract.get_layer('extract_features').output)

        encode_dev = Embedder.predict([text_dev])
        if phase == 'both':
            encode_train = Embedder.predict([text])
    
    if encode_dev is not None:
        np.save('../../data/flow_dev', encode_dev)
    if encode_train is not None:
        np.save('../../data/flow_train', encode_train)
