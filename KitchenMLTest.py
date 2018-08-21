# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:14:06 2018

@author: Admin
"""

#from pandas import Series
#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.graphics.tsaplots import plot_pacf
#from matplotlib import pyplot
#from pandas import DataFrame
#from pandas import read_csv
#from pandas import datetime
#
#def parser(x):
#	return datetime.strptime(x, '%Y-%m-%d')
#
#series = read_csv('recom_train.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
##print(series.head())
#
#pyplot.figure(figsize=(30,10))
#pyplot.subplot(211)
#plot_acf(series, ax=pyplot.gca())
#pyplot.subplot(212)
#plot_pacf(series, ax=pyplot.gca())
#pyplot.show()
import warnings
#from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import datetime
from pandas import read_csv
 
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
 
# load dataset
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

series = read_csv('data/recom_train.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# evaluate parameters
p_values = range(0,13)
d_values = range(0, 4)
q_values = range(0, 13)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)