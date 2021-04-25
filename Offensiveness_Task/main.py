#%%
import argparse, sys, os, numpy as np
from models.lm_finetunning import train_language_model
from models.bert_based import bert_based
from models.xlnet import xlnet_process
from Sentiment_Flow.generator import generate_sentiment_flow

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-mode', metavar='mode', help='Select Prediction Mode', default=['reps','fcnn', 'ridge'])
  parser.add_argument('-dp', metavar='data_path', help='Select Data Path')
  parser.add_argument('-dt', metavar='data_path_dev', help='Select Data Path for Development Set')
  

if __name__ == '__main__':

  parameters = check_params(sys.argv[1:])

  mode = parameters.mode
  data_path = parameters.pd
  data_path_dev = parameters.dt

  if mode == 'reps':

    ''' 
      Train Transformers Models for Getting Best Representations. Generate Representations
    '''

    for i in ['bt', 'rob']:
      train_language_model(data_path, i) #'../data/data_for_mlm.csv'
      train_language_model(data_path, i)
      bert_based(i, data_path, data_path_dev, phase='both', splits=5)
    xlnet_process(data_path, data_path_dev, phase='both', splits=5)
    generate_sentiment_flow(data_path, data_path_dev, phase='both')


    
