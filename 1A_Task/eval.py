"""
USAGE: eval.py [reference] [predicted]
"""

import argparse, sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score


def eval(y_test, y_predicted):    

	precision, recall, fscore, _ = score(y_test, y_predicted)
	print('\n     {0}   {1}'.format("0","1"))
	print('P: {}'.format(precision))
	print('R: {}'.format(recall))
	print('F: {}'.format(fscore))

	mprecision, mrecall, mfscore, _ = score(y_test, y_predicted, average='macro')
	print('\n MACRO-AVG')
	print('P: {}'.format(mprecision))
	print('R: {}'.format(mrecall))
	print('F: {}'.format(mfscore))
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('reference', help='reference set')
	parser.add_argument('predicted', help='system output')
	args = parser.parse_args() 

	y_test = pd.read_csv(sys.argv[1])['is_humor']
	y_predicted = pd.read_csv(sys.argv[2])['is_humor']

	y_test = y_test.to_numpy().tolist()
	y_predicted = y_predicted.to_numpy().tolist()
		
	eval(y_test, y_predicted)
			
	  
  