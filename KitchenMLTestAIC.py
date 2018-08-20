from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA


def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

series = read_csv('data/recom_train.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# fit model
print("Minimum AIC Value")
model = ARIMA(series, order=(6,2,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
print("\n")

print("Maximum AIC Value")
model = ARIMA(series, order=(1,2,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())