#%%
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sys.path.append('./feature_extractor/')
import keras_bert, os, pandas as pd, numpy as np
import tensorflow as tf, numpy as np
import keras as K
from utils import load_data, masked_mse, f1, get_splits_for_val, trucncated_under_sampling
import tensorflow as tf
import keras as K
from utils_feature import convert_lines
from matplotlib import pyplot as plt
from keras_self_attention import SeqSelfAttention

def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf


def FNNATT_Model(inshape):

    e1 = K.layers.Input(inshape, name='input_bert')
    e2 = K.layers.Input(inshape, name='input_roberta')
    e3 = K.layers.Input(inshape, name='input_xln')
    e4 = K.layers.Input(inshape, name='input_features')
    
    E1 = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(e1)
    E2 = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(e2)
    E3 = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(e3)
    E4 = K.layers.Lambda(lambda x : K.backend.expand_dims(x, 2))(e4)
    
    #quite el encode de features porque no da en tamanno
    extract = K.layers.concatenate([E1, E2, E3, E4], axis = 2, name='Concatenate_BERT_output')
    extract = K.layers.Permute((2, 1))(extract)
    # print(extract)
    # extract = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(extract)
    extract = K.layers.LSTM(units=94, name='simple_lstm')(extract)
    # extract = K.layers.RNN(cell=MinimalRNNCell(units=100))(extract)
    # extract = K.backend.mean(extract, axis=1)
    print(extract.shape)
    X = K.layers.Dense(64, activation=gelu, name = 'extract_features')(extract)

    X = K.layers.BatchNormalization(name='norm1')(X)
    # X = K.layers.Dense(32, activation='relu', name = 'encoder_layer')(X)

    output_offense = K.layers.Dense(2, activation='softmax', name='output_offense')(X)
    output_offensiveness = K.layers.Dense(1, activation=gelu, name='output_offensiveness')(X)


    return K.Model(inputs=[e1, e2, e3, e4], outputs=[ output_offense, output_offensiveness])

def compute_mean_squared_error(y, yhat):
    loss = yhat - offensiveness_dev
    loss = np.sqrt(np.sum(loss**2)/len(offensiveness_dev))
    return loss


def root_mean_squared_error(y_true, y_pred):
        from keras import backend as bk
        m = bk.sum(bk.clip(y_true + 5, 0, 1))
        return bk.sqrt(bk.sum(bk.pow(y_true - y_pred, 2))/(m+bk.epsilon()))
        

def Minkowski_loss(y_true, y_pred):
    return K.backend.mean(K.backend.pow(K.backend.abs(y_pred-y_true), 1.4))

losses = {'output_offense': K.losses.categorical_crossentropy, 'output_offensiveness':Minkowski_loss }
metrics = {'output_offense': f1, 'output_offensiveness':root_mean_squared_error }


DATA_PATH = '../data/train.csv'
text, is_humor, humor_rating, controversy, offensiveness = load_data(DATA_PATH)
text_dev, is_humor_dev, humor_rating_dev, controversy_dev, offensiveness_dev = load_data('../data/dev.csv')

bert_encodes = '../data/bertweet_'
roberta_encodes = '../data/roberta_'
xlnet_encodes = '../data/xlnet_'
features_path = '../data/features/encode_f_'

def load_encodes(phase):
    z = (np.load(bert_encodes + phase + '_encode.npy'), np.load(roberta_encodes + phase + '_encode.npy'),
         np.load(xlnet_encodes + phase + '_encode.npy'), np.load(features_path + phase + '.npy'))
    return z

textb, textr, textx, features = load_encodes('train')
textdevb, textdevr, textdevx, features_dev = load_encodes('dev')
texttestb, texttestr, texttestx, features_test = load_encodes('test')

splits = 1

is_offense = np.array([0 if i == 0 else 1 for i in offensiveness])
is_offense_dev = np.array([0 if i == 0 else 1 for i in offensiveness_dev])

iddev = pd.read_csv('../data/public_dev.csv').to_numpy()[:, 0]
id_test = pd.read_csv('../data/public_test.csv').to_numpy()[:, 0]

preds = {'humor':np.zeros((len(iddev), )), 'rating':np.zeros((len(iddev), )), 'cont':np.zeros((len(iddev), )),'off':np.zeros((len(iddev), )),}
preds_test = {'humor':np.zeros((len(id_test), )), 'rating':np.zeros((len(id_test), )), 'cont':np.zeros((len(id_test), )),'off':np.zeros((len(id_test), )),}

tf.random.set_seed(21)
np.random.seed(21)
real_plot = []
pred_plot = []
 

hist = plt
splits = 1

filepath = "weights_fnn.h5"

opt = K.optimizers.RMSprop(lr=1e-3, decay = 1e-3)
ATT_FNN = FNNATT_Model(textb[0].shape)
# ATT_FNN.summary(line_length=170)
ATT_FNN.compile(optimizer=opt, loss=losses, metrics=metrics)

checkpointer = K.callbacks.ModelCheckpoint(filepath, monitor='val_output_offensiveness_root_mean_squared_error', mode='min', verbose=1, save_best_only=True, save_weights_only =True)


history = ATT_FNN.fit([textb , textr , textx , features ] ,[K.utils.to_categorical(is_offense ), offensiveness ],
            validation_data=([textdevb, textdevr, textdevx, features_dev],
                            [K.utils.to_categorical(is_offense_dev), offensiveness_dev]),
            batch_size=32, epochs=15, callbacks=[checkpointer], verbose=2)

ATT_FNN.load_weights(filepath)
ATT_FNN.evaluate([textdevb, textdevr, textdevx, features_dev],
                            [ K.utils.to_categorical(is_offense_dev), offensiveness_dev], verbose=2)

pred_plot += list(ATT_FNN.predict([textdevb, textdevr, textdevx, features_dev])[1])
real_plot += list(offensiveness_dev)


##---------------------vealuate dev
z = ATT_FNN.predict([textdevb, textdevr, textdevx, features_dev])[1]
# preds['rating'] += np.array(z).reshape(-1)
preds['off'] = np.array(z)

#--------------test
z = ATT_FNN.predict([texttestb, texttestr, texttestx, features_test])[1]
# preds_test['rating'] += np.array(z).reshape(-1)
preds_test['off'] += np.array(z).reshape(-1)

# preds['humor'] = np.array(np.round(preds['humor']/splits), dtype=np.int)
preds['off'] = np.round(np.clip(preds['off']/splits, 0, 5), decimals=2)
# preds['rating'] =  np.round(np.clip(preds['rating']/splits, 0, 5), decimals=2)
# preds['cont'] = np.array(np.round(preds['cont']/splits), dtype=np.int)

# preds_test['humor'] = np.array(np.round(preds_test['humor']/splits), dtype=np.int)
preds_test['off'] = np.round(np.clip(preds_test['off']/splits, 0, 5), decimals=2)
# preds_test['rating'] =   np.round(np.clip(preds_test['rating']/splits, 0, 5), decimals=2)
# preds_test['cont'] = np.array(np.round(preds_test['cont']/splits), dtype=np.int)


dictionary = {'id': iddev,  'offense_rating':[i[0] for i in preds['off']]}  
df = pd.DataFrame(dictionary) 
df.to_csv('../att_fnn_dev.csv')
print('Evaluation Done Dev!!') 

# dictionary = {'id': id_test, 'offense_rating':preds_test['off']}  
# df = pd.DataFrame(dictionary) 
# df.to_csv('../best sumbissions/send/noo_att_rnn_test.csv')
# print('Evaluation Done Test!!') 

