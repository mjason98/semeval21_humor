import pandas as pd, numpy as np, os, sys
import keras as K, tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMaskedLM
from matplotlib import pyplot as plt
from Optimizers.RMS_Nest.NadamW import NadamW
from Optimizers.AdamLRM.adamlrm import AdamLRM
from utils import set_lt_multipliers


def tokenize(example, max_seq_length, tokenizer):

    max_seq_length-=2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
      
      tokens_a = tokenizer.tokenize(example[i])
      if len(tokens_a ) > max_seq_length:
          tokens_a = tokens_a[:max_seq_length]
          longer += 1
      all_tokens.append(np.array(tokenizer.convert_tokens_to_ids(["<s>" ] +tokens_a +["</s>"] )))
    return np.array(all_tokens)

def random_masking(examples, mask, mlindex = 0.15, ):


  for i in range(len(examples)):
    index_for_mask = np.random.permutation(len(examples[i]))[:int(np.ceil(len(examples[i])*mlindex))]
    examples[i][index_for_mask] = mask
  return examples

def padding(items, max_seq_length):

  padded = []
  for i in range(len(items)):
    one_token = np.array(list(items[i]) + [tokenizer.pad_token_id]*(max_seq_length - len(items[i])))
    padded.append(one_token)
  return np.array(padded)

def preprocessing(raw_texts, max_seq_length, tokenizer):

  example = tokenize(raw_texts, max_seq_length, tokenizer)
  labels = tokenize(raw_texts, max_seq_length, tokenizer)
  example = random_masking(example, tokenizer.mask_token_id)

  example = padding(example, max_seq_length)
  labels = padding(labels, max_seq_length)
  
  return example, labels

def splits_batch(batch_size, m):
  perm = np.random.permutation(m)

  splits = []
  for i in range(int(m/batch_size)):
    splits.append(perm[i*batch_size:(i+1)*batch_size])
  if m%batch_size != 0:
    splits.append(perm[m*batch_size:])
  return splits


def train_language_model(model_name, data):

  if model_name=='bt':
    pt_path = "vinai/bertweet-base"
  elif model_name=='rob':
    pt_path = "roberta-base"

  tokenizer = AutoTokenizer.from_pretrained(pt_path) 
  model =  TFAutoModelForMaskedLM.from_pretrained(pt_path, return_dict=True)

  text = pd.read_csv(data).to_numpy()[:, 1]
  text, labels = preprocessing(text, 70, tokenizer)

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(1e-5)

  batch_size = 32
  history = []
  for i in range(4):

    batch_splits = splits_batch(batch_size, len(text))
    loss = 0
    perc = 0

    for j in range(len(batch_splits)):
      with tf.GradientTape() as g:

          if j*100.0/len(batch_splits) - perc  >= 1:
            perc = j*100.0/len(batch_splits)
            print('\r Epoch:{} setp {} of {}. {}%'.format(i+1, j, len(batch_splits), np.round(perc, decimals=2)), end="")

          out = model(text[batch_splits[j],:])
          loss_value = loss_object(y_true=labels[batch_splits[j],:], y_pred=out.logits)
      gradients = g.gradient(loss_value, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      loss += loss_value.numpy()/len(batch_splits)
    history.append(loss)
    print('\repoch: {} Loss: {}'.format( i+1, loss))

  plt.plot(history)
  plt.legend('train', loc='upper left')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.show()
  os.system('mkdir -p ../data/lm_fine_tunning')
  model.save_pretrained('../data/lm_fine_tunning')