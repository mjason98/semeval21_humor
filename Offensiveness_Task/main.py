#%%
import argparse, sys, os, numpy as np
from models.lm_finetunning import train_language_model


def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-mode', metavar='mode', help='Select Prediction Mode', default=['reps','fcnn', 'ridge'])
  parser.add_argument('-dp', metavar='data_path', help='Select Data Path')
  

if __name__ == '__main__':

  parameters = check_params(sys.argv[1:])

  mode = parameters.mode
  data_path = parameters.pd

  if mode == 'reps':

    ''' 
      Train Transformers Models for Getting Best Representations. Generate Representations
    '''

    train_language_model(data_path, 'bt') #'../data/data_for_mlm.csv'
    train_language_model(data_path, 'rob')
    
