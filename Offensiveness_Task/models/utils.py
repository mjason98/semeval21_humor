import pandas as pd
import numpy as np
import math, os
import keras as K
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# import pyfreeling

stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]


# from keras.legacy import interfaces
# import keras.backend as bk
# from tensorflow.keras.optimizers import Optimizer

def load_data(DATA_FILE):

    file = pd.read_csv(DATA_FILE)
    file = file.to_numpy()
    
    text = file[:,1]
    is_humor = np.array(file[:,2])    #binary
    humor_rating = np.array(file[:,3], dtype = np.float64)
    controversy = np.array(file[:,4]) #binary
    offensiveness = np.array(file[:,5], dtype = np.float64)

    for i in range(len(humor_rating)):
        if math.isnan(humor_rating[i]):
            humor_rating[i] = 0
        if math.isnan(controversy[i]):
            controversy[i] = 0
    # humor_rating = np.where(np.isnan(humor_rating), humor_rating, np.zeros_like(humor_rating))
    # controversy = np.where(str(controversy) == 'nan', np.zeros_like(controversy), controversy)

    return text, is_humor, humor_rating, controversy, offensiveness

def load_data_hateval(DATA_FILE):

    file = pd.read_csv(DATA_FILE)
    file = file.to_numpy()
    
    text = file[:,1]
    hate = file[:, 2]
    agresivness = file[:, 4]

    return text, hate, agresivness

def load_data_sarcasm(DATA_FILE):

    file = pd.read_csv(DATA_FILE)
    file = file.to_numpy()
    
    text = file[:,2]
    sarcasm = np.array(file[:,1], dtype = np.float64) 

    return text, sarcasm

def load_data_offenseval(DATA_FILE):

    file = pd.read_csv('../data/offenseval-training-v1.tsv', sep='\t')
    file = file.to_numpy()
    text = file[:,1]
    offensive = np.array([1 if 'OFF' in i else 0 for i in file[:,2]])
    index_n0off = np.array([i for i in range(len(offensive)) if offensive[i] == 0])
    index_n0off = index_n0off[np.random.permutation(len(index_n0off))]
    indexs = list(set([i for i in range(len(offensive))]) - set(index_n0off[np.sum(offensive) + 1000:]))

    text = text[indexs]
    offensive = offensive[indexs]
    return text, np.array(offensive)

def convert_lines(example, max_seq_length ,tokenizer):
    max_seq_length-=2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
        tokens_a = tokenizer.tokenize(example[i])
        if len(tokens_a ) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids([tokenizer.bos_token ] +tokens_a +[tokenizer.eos_token] ) +[0] * \
                    (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
        
    print(longer)
    return np.array(all_tokens, dtype = np.int32)


def f1(y_true, y_pred):
        from keras import backend as bk
        def recall(y_true, y_pred):
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = bk.sum(bk.round(bk.clip(y_true * y_pred, 0, 1)))
            possible_positives = bk.sum(bk.round(bk.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + bk.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = bk.sum(bk.round(bk.clip(y_true * y_pred, 0, 1)))
            predicted_positives = bk.sum(bk.round(bk.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + bk.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+bk.epsilon()))
        
def Evaluate(features_dev, mymodel, path_data, maxlen, namefile, tokenizer):
    file = pd.read_csv(path_data)
    file = file.to_numpy()

    id = file[:, 0]
    text = file[:,1]
    text = convert_lines(text, maxlen, tokenizer)
    # mask_ = np.ones((text.shape[0], maxlen))

    z = mymodel.predict([text, features_dev])
    # is_humorans = np.argmax(z[0], axis = 1)
    # humor_ratingans = np.round(np.clip(z[0].reshape(-1), 0, 5), decimals=2)
    # controversyans = np.argmax(z[2], axis = 1)
    offensivenessans = np.round(np.clip(z[1] , 0, 5).reshape(-1), decimals=2)

    # dictionary = {'id': id, 'is_humor': is_humorans, 'humor_rating':humor_ratingans, 'humor_controversy':	controversyans, 'offense_rating':offensivenessans}  
    dictionary = {'id': id, 'offense_rating':offensivenessans}  
    df = pd.DataFrame(dictionary) 
    
    df.to_csv('/content/drive/MyDrive/Semeval/data/' + namefile + '.csv')
    print('Evaluation Done!!') 

def Save_Encode(features, model, path_data, maxlen, tokenizer):
    file = pd.read_csv(path_data)
    file = file.to_numpy()

    id = file[:, 0]
    text = file[:,1]
    text = convert_lines(text, maxlen, tokenizer)
    mask_ = np.ones((text.shape[0], maxlen))

    layer_output = model.get_layer('encoder_layer').output
    Embedder = K.models.Model(inputs=model.input, outputs=layer_output)
    enc = Embedder.predict([text, features])
    np.save(path_data[:-4] + '_encode', enc)

def Save_Encodexln( model, path_data, maxlen, tokenizer):
    file = pd.read_csv(path_data)
    file = file.to_numpy()

    id = file[:, 0]
    text = file[:,1]
    text = convert_lines(text, maxlen, tokenizer)
     

    layer_output = model.get_layer('encoder_layer').output
    Embedder = K.models.Model(inputs=model.input, outputs=layer_output)
    enc = Embedder.predict([text ])
    np.save(path_data[:-4] + '_encode', enc)

def Merge_submitions(file, namefile):
  humor = pd.read_csv(file[0], usecols=['is_humor']).to_numpy()[:,0]
  humor_rating = pd.read_csv(file[1], usecols=['humor_rating']).to_numpy()[:,0]
  controversy = pd.read_csv(file[2], usecols=['humor_controversy'])
  offense_rating = pd.read_csv(file[3], usecols=['offense_rating'])
  
  dictionary = {'id': id, 'is_humor': humor, 'humor_rating':humor_rating, 'humor_controversy':	controversy, 'offense_rating':offense_rating}  
  df = pd.DataFrame(dictionary) 
  
  df.to_csv('/content/drive/MyDrive/Semeval/data/' + namefile + '.csv')
  print('Evaluation Done!!') 



def get_splits_for_val(offensiveness, splits):
  values = {}
  for i in range(len(offensiveness)):
    k = offensiveness[i]
    if values.get(k) is None:
      values[k] = [i]
    else: values[k].append(i)

  data_val = [ set() for i in range(splits)]
  for i in values.keys():

    scaques = np.random.permutation(splits)
    if len(values[i]) < splits:
      for j in range(len(values[i])):
        data_val[scaques[j]].add(values[i][j])
      continue

    val_segm = int(len(values[i])/splits)
    for j in range(splits):
      t = scaques[j]
      data_val[t] |= set(values[i][j*val_segm:(j+1)*val_segm])

    if len(values[i])%splits != 0:
      data_val[scaques[splits-1]] |= set(values[i][splits*val_segm:])
  return data_val

def trucncated_under_sampling(offensiveness, treshold):
  index_used = {}
  for i in range(len(offensiveness)):
    k = offensiveness[i]
    if index_used.get(k) is None:
      index_used[k] = [i]
    else: index_used[k].append(i)

  for i in index_used.keys():
    index_used[i] = np.array(index_used[i])

  allindex = []
  for i in index_used.keys():
    if index_used[i].shape[0] > treshold:
      index_used[i] = index_used[i][np.random.permutation(index_used[i].shape[0])]
      index_used[i] = index_used[i][:treshold]
    allindex += list(index_used[i])
  return np.array(allindex)

def Minkowski_masked_loss(y_true, y_pred):
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx())
  return K.backend.mean(K.backend.pow(K.backend.abs(y_pred*mask-y_true*mask), 1.4))

def Minkowski_loss(y_true, y_pred):
    return K.backend.mean(K.backend.pow(K.backend.abs(y_pred-y_true), 1.2))

def root_mean_squared_error(y_true, y_pred):
    return K.backend.sqrt(K.losses.mean_squared_error(y_true, y_pred))

def masked_root_mean_squared_error(y_true, y_pred):
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx())
  return K.backend.sqrt(K.losses.mean_squared_error(y_true*mask, y_pred*mask))

def masked_mse(y_true, y_pred):
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx())
  return K.losses.mean_squared_error(y_true*mask, y_pred*mask)

def masked_categortical_crossentropy(y_true, y_pred):
  
  mask = K.backend.cast(K.backend.not_equal(y_true, -1), K.backend.floatx()) 
  return K.losses.categorical_crossentropy(y_true*mask, y_pred*mask)

class MaskedCategoricalAccuracy(tf.keras.metrics.CategoricalAccuracy):
    
    def update_state(self, y_true, y_pred, sample_weight=None):
      mask = tf.cast(tf.math.reduce_any(tf.not_equal(y_true, -1), axis=1), tf.float32) 
      return super().update_state(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask), sample_weight,)
     

def set_lt_multipliers(start, model):

  coef_learning = {}
  
  ml = start 

  x = [i.name for i in model.layers[1].layers[0].encoder.layer]
  # x.reverse()
  for i in x:
    coef_learning[i] = ml
    ml += 0.1

    x = [ i.name for i in model.layers][3:]
  for i in x:
    coef_learning[i] = ml
  return coef_learning

def set_lt_multipliers_gpt2(start, model):

  coef_learning = {}
  
  ml = start 

  x = ['h_._'+str(i) for i in range(12)]
  # x.reverse()
  for i in x:
    coef_learning[i] = ml
    ml += 0.1

    x = [ i.name for i in model.layers][3:]
  for i in x:
    coef_learning[i] = ml
  return coef_learning

def set_lt_multipliers_xlnet(start, model):

  coef_learning = {}
  ml = start
  for i in model.layers[1].transformer.layer:
      ml += 0.1
      coef_learning[i.name] = ml

  x = [ i.name for i in model.layers][3:]
  ml += 0.1

  for i in x:
    coef_learning[i] = ml
  return coef_learning

def set_lt_multipliers_for_Bert(start, model):
    coef_learning = {}
    ml = 1
    x = [ i.name for i in model.layers][3:]
    x.reverse()

    for i in x:
      coef_learning[i] = ml
    
    ml *= 0.5
    x = [i.name for i in model.layers[1].layers[0].encoder.layer]
    x.reverse()

    for i in x:
      coef_learning[i.name] = ml
      ml *= 0.5

    return coef_learning

def euclidean(x, y):
    z = x-y
    z **= 2
    return np.sqrt(np.sum(z))
class GLC():

    def __init__(self, nodes=np.array([]), umbral=0.75):

        self.nodes = nodes
        self.__cluster_index = None
        self.__neighbor = None
        self.clusters = 0
        self.__umbral = umbral
        self.centroids = None

        self.dist_mean = [0 for i in range(len(self.nodes))]

    def compute_euclidean(self, x, y):
        z = x-y
        z **= 2
        return np.sqrt(np.sum(z))


    def bfs(self, node, index):
        self.__cluster_index[node] = index
        q = []
        q.append(node)

        while len(q) > 0:
            node = q.pop()

            for new_node in self.__neighbor[node]:
                if self.__cluster_index[new_node] > -1:
                    continue
                self.__cluster_index[new_node] = index
                q.append(new_node)

    def find_centroids(self, type):

        self.__neighbor = {}
        tmp = []
        weights = []

        print(self.__umbral, len(self.nodes))
        for i in range(len(self.nodes)):
            weights.append([])            
            for j in range(i+1, len(self.nodes)):
                weights[-1].append(self.compute_euclidean(self.nodes[i], self.nodes[j]))
                self.dist_mean[i] += weights[-1][-1]
                self.dist_mean[j] += weights[-1][-1] 
        tmp = []

        for i in weights:
            tmp += i
        
        tmp.sort()
        
        u = tmp[int(self.__umbral*len(tmp))]

        for i in range(len(self.nodes)):
            self.dist_mean[i] /= len(self.nodes)
            B_values = u
            self.__neighbor[i] = []
            k = 0
            for j in range(i+1, len(self.nodes)):
                # X = self.compute_euclidean(self.nodes[i], self.nodes[j])
                if abs(weights[i][k] - B_values) < 1e-9:
                    self.__neighbor[i].append(j)
                elif B_values - weights[i][k] > 1e-9:
                    B_values = weights[i][k]
                    self.__neighbor[i].clear()
                    self.__neighbor[i].append(j)
                k += 1
        del tmp
        del weights

        for i in range(len(self.nodes)):
            for j in self.__neighbor[i]:
                self.__neighbor[j].append(i)

        for i in range(len(self.nodes)):
            self.__neighbor[i] = set(self.__neighbor[i])

        self.__cluster_index = [-1 for i in range(len(self.nodes))]

        self.clusters = 0
        for i in range(len(self.nodes)):
            if self.__cluster_index[i] == -1:
                self.bfs(i, self.clusters)
                self.clusters += 1

        centroids = [np.array([-1, -1]) for i in set(self.__cluster_index)]
        if type == 'dg':
            for i in range(len(self.nodes)):
                if centroids[self.__cluster_index[i]][1] < len(self.__neighbor[i]):
                    centroids[self.__cluster_index[i]][1] = len(self.__neighbor[i])
                    centroids[self.__cluster_index[i]][0] = i
        
        elif type == 'mean':
            
            tmp = [(self.dist_mean[i], i) for i in range(len(self.nodes))]
            tmp.sort()
            for i in tmp:
                if centroids[self.__cluster_index[i[1]]][0] == -1:
                    centroids[self.__cluster_index[i[1]]][0] = i[1] 
            del tmp
        # print(centroids)
        ret = np.array([i[0] for i in centroids])
        return ret

def class_partition(X, U, clas, labels, proto):

    index = [i for i in range(len(labels)) if labels[i] == clas]
    X1 = X[index]
    index = np.array(index)

    clus = GLC(X1, U)
    prototype = clus.find_centroids(proto)


    all = set([i for i in range(len(X1))]) - set(prototype)
    X1 = index[list(all)]
    prototype = index[prototype]

    return np.array(prototype), np.array(X1)

def make_pairs(X, clas, clas_protot, opositeclass, K):

    Anchor = []
    VS = []
    labels = []

    for i in clas:
        op = [(euclidean(X[i], X[j]), j) for j in opositeclass]
        op.sort()
        for j in range(K):
            Anchor.append(i)
            VS.append(op[j][1])
            labels.append(0)
       

        x = list(np.random.permutation(len(clas_protot)))
        for j in range(K):
            Anchor.append(i)
            VS.append(clas_protot[x[j]])
            labels.append(1)

    return np.array(Anchor), np.array(VS), np.array(labels)

def plot_classes(Z, X, X1, PT1, met, proto):

    # method = {'tsne':TSNE(n_components=2), 'pca':PCA(n_components=2)}
    # feat_sel = method[met]
    # Z = feat_sel.fit_transform(X)

    A1 = Z[X1]
    P1 = Z[PT1]

    fig = plt.figure()
    # pos = fig.add_subplot(1, 2, 1)
    # neg = fig.add_subplot(1, 2, 2)
    colors = ['b', 'g', 'r', 'y', 'w']
    plt.scatter(A1[:,0], A1[:,1], c = 'b', label = 'Pos')
    plt.scatter(P1[:,0], P1[:,1], c = 'r', label = 'Prot pos ' + str(len(PT1)))
    # pos.legend(loc=1)

    # A0 = Z[X0]
    # P0 = Z[PT0]
    # plt.scatter(A0[:,0], A0[:,1], c = 'y', label = 'Negatives', alpha=1)
    # plt.scatter(P0[:,0], P0[:,1], c = 'g', label = 'Prot Nega ' + str(len(PT0)), alpha=1)

    plt.legend(loc=1)
    plt.savefig('prototypes_' + proto, orientation='landscape')
    plt.show()

def Merge_submitions(file, namefile):
  
  idx = pd.read_csv(file[0], usecols=['id']).to_numpy()[:,0]
  is_humorans = pd.read_csv(file[0], usecols=['is_humor']).to_numpy()[:,0]
  humor_ratingans = pd.read_csv(file[1], usecols=['humor_rating']).to_numpy()[:,0]
  controversyans = pd.read_csv(file[2], usecols=['humor_controversy']).to_numpy()[:,0]
  offensivenessans = pd.read_csv(file[3], usecols=['offense_rating']).to_numpy()[:,0]

  dictionary = {'id': idx, 'is_humor': is_humorans, 'humor_rating':humor_ratingans, 'humor_controversy':	controversyans, 'offense_rating':offensivenessans}  
  df = pd.DataFrame(dictionary) 
  
  df.to_csv( namefile + '.csv')
  print('Merge Done!!') 


# file = ['submition/is_humor.csv', 'submition/controv_rating.csv', 'submition/controv_rating.csv', 'submition/offensiveness.csv']
# Merge_submitions(file, 'Reg+Clas(logcosh)')
# %%
# def load_lemmatized(DATA_PATH):

#     def load_data(data_path):
#         file = pd.read_csv(data_path)
#         file = file.to_numpy()

#         text = file[:,1]
#         return text

#     def Preprocess(texts):
#         print('Freeling Tokenizing ', '0 %\r', end="")
#         if "FREELINGDIR" not in os.environ:
#             os.environ["FREELINGDIR"] = "/usr/local"
#         DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

#         pyfreeling.util_init_locale("default")
#         LANG = "en"
#         op = pyfreeling.maco_options(LANG)
#         op.set_data_files("",
#                           DATA + "common/punct.dat",
#                           DATA + LANG + "/dicc.src",
#                           DATA + LANG + "/afixos.dat",
#                           "",
#                           DATA + LANG + "/locucions.dat",
#                           DATA + LANG + "/np.dat",
#                           DATA + LANG + "/quantities.dat",
#                           DATA + LANG + "/probabilitats.dat")

#         tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
#         sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
#         mf = pyfreeling.maco(op)
#         mf.set_active_options(False, True, True, True,  # select which among created
#                               True, True, True, True,  # submodules are to be used.
#                               True, True, False, True)  # default: all created submodules are used

#         tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
#         sen = pyfreeling.senses(DATA + LANG + "/senses.dat")

#         done = 0
#         perc = 0
#         top = len(text)
#         cont = 0
#         for i in range(len(text)):

#             done += 1
#             z = done / top
#             z *= 100
#             z = int(z)
#             if z - perc >= 1:
#                 perc = z
#                 print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

#             x = text[i] + '.'
#             l = tk.tokenize(x)
#             #         sid=sp.open_session();
#             ls = sp.split(l)

#             ls = mf.analyze(ls)
#             ls = tg.analyze(ls)
#             ls = sen.analyze(ls)

#             setnteces = ""
#             for s in ls:
#                 cont  += 1
#                 ora = s.get_words()
                
#                 for k in range(len(ora)):
#                     x = ora[k].get_lemma().lower()
#                     if ora[k].get_tag() != 'W' and ora[k].get_tag() != 'Z' and ora[k].get_tag()[0] != 'F' and stop_words.count(x) == 0 and stop_words.count(ora[k].get_form().lower()) == 0:
#                         setnteces += x + ' '
#             text[i] = setnteces[:-1]

#         print('Freeling Tokenizing ok        ')
#         return text

#     text = load_data(DATA_PATH)
#     text = Preprocess(text)
#     return text

def convert_lines_for_BERT(example, max_seq_length ,tokenizer):
    max_seq_length-=2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
        tokens_a = tokenizer.tokenize(example[i])
        if len(tokens_a ) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]" ] +tokens_a +["[SEP]"] ) +[0] * \
                    (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
        
    print(longer)
    return np.array(all_tokens)

def Evaluate_BERT(mymodel, path_data, maxlen, tokenizer):
    file = pd.read_csv(path_data)
    file = file.to_numpy()

    id = file[:, 0]
    text = file[:,1]
    text = convert_lines_for_BERT(text, maxlen, tokenizer)
    seg_ = np.zeros((text.shape[0], maxlen))
    mask_ = np.ones((text.shape[0], maxlen))
    z = mymodel.predict([text, seg_, mask_])
    is_humorans = np.argmax(z[0], axis = 1)
    humor_ratingans = np.clip(z[1], 0, 5).reshape(-1)
    controversyans = np.argmax(z[2], axis = 1)
    offensivenessans = np.clip(z[3], 0, 5).reshape(-1)

    dictionary = {'id': id, 'is_humor': is_humorans, 'humor_rating':humor_ratingans, 'humor_controversy':	controversyans, 'offense_rating':offensivenessans}  
    df = pd.DataFrame(dictionary) 
    
    df.to_csv('file1.csv') 