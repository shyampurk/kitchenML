from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import sys
import csv
import os
import os.path


container_capacity = 25
historical_pizza_sales_data = 'data/recom_train.csv'

model = None
model_fit = None

# parser - Function to parse the date from the cv file
#
#
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

# generateRecommendation - recommendation logic for predicting the required amount of Mozzarella cheese
#
#
def generateRecommendation():
    #Read historical sales data
    series = read_csv(historical_pizza_sales_data, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    series.plot()
    pyplot.show()
    
    #read the historical sales data and capture last 55 sample
    with open(historical_pizza_sales_data, 'r', newline='') as csvfile:
         inde = csv.reader(csvfile)
         times = []
         for i, row in enumerate(inde):
                 times.append(row[0])
        
         last_55_sample_time=[]
         last_55_sample_time=times[-55:]
         #print(tim)
    
   
    #Split the data into training and tesitng set, training set to constitute 90% of samples
    X = series.values
    train_size = int(len(X) * 0.90)
    train, test = X[0:train_size], X[train_size:]
    
    #Copy the training set into a variable history for running the loop
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
         #Build the ARMIA model with the series with random values for p,d and q
         #p - Number of lag observations included in model (AR)
         #d - Number of times the raw observations are differenced (I)
         #q - Size of moving average window (MA)
        model = ARIMA(history, order=(6,2,0))
        model_fit = model.fit(disp=0)
        
        #Get the next forcasted value for pizza sales
        output = model_fit.forecast()
        yhat = output[0]
        
        #Add the forecasted value to prediction list
        predictions.append(yhat)
        
        #Get the next observed value from test data
        obs = test[t]
        
        #Append test data to history for the next loop iteration
        history.append(obs)
        
    g=len(predictions)
    #print(g)
    #print(predictions)
    
    #Convert the predictions array to list
    convert=[]
    for i in range(g) :
        con=np.array(predictions[i]).tolist()
        convert.append(con)
       # print(convert)
    
    #Make a single list of predicted values
    flattened=[]
    for sublist in convert:
        for val in sublist:
            flattened.append(val)
    #print(flattened)
    
    #Write the predicted values in a csv file aligning with the last 55 sample times
    # (which is same as the testing set)
    if(os.path.isfile('prediction.csv')):
        os.remove('prediction.csv')
    with open('prediction.csv', 'a',newline='') as myFile:
        writer=csv.writer(myFile, skipinitialspace=False,delimiter = ',')
        #for i in range(g) :
        writer.writerows(zip(last_55_sample_time, flattened))
            #writer.writerow(['', predictions1[i]])
    myFile.close()
    
    
    prediction_series = read_csv('prediction.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    prediction_series.plot(color='red')
    pyplot.show()
    
    print('\n')
    model = ARIMA(history, order=(6,2,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    print('Predicted=%f' % (yhat))
    
    #Multiply the predicted pizza sales figure with 50, indicating 50gms for each pizza
    n=yhat*50
    if(n>0):
        #Divide by 1000 to get the value in Kgs for cheese requirement
        fi=n/1000;
        print("Recommended Order Quantity is %f Kgs" % fi)
    else:
        print("No Recommendation, Quantity is sufficient!")
    mse = mean_squared_error(test, predictions)
    rmse = math.sqrt(mse)
    print('MSE %.3f' % mse) 
    print('RMSE: %.3f' % rmse)

    forecast_errors = [abs(((test[i]-predictions[i])*100)/(test[i])) for i in range(len(test))]
    error = sum(forecast_errors) * 1.0/len(test)
    print('Error percentage (MAPE): %f' % error)
    #mae = mean_absolute_error(test, predictions)
    #print('MAE: %f' % mae)

    
    
                 
# Load the daily consumption data file for cheese.
# p - stannds for positive that means the the prediction will be positive
# n - stands for negative that means the prediction will be negative
# c - stands for constant that means the prediction will be nearly constant

a=sys.argv[1]
print(int(a))

if a=="1":
    df = pd.read_csv("data/noti_p.csv")
    print("Executing Scenario 1 - Stock will last almost till the end of day")
elif a=="2":
    df = pd.read_csv("data/noti_n.csv")
    print("Executing Scenario 2 - Stock will not last till the end of day")
elif a=="3":
    df = pd.read_csv("data/noti_c.csv") 
    print("Executing Scenario 3 - Stock will be surplus at the end of the day")
else:
    print("Invalid Scenario. Enter Valid argumet - 1,2 or 3 for scenario no.")
    sys.exit()
    
    

    
Y = df['Kgs']
X = df['time']
 
#Reshape the data by ignoring the first row
X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)
 
#Take the data samples from 1st to 7th ( ignore the zreoth sample as it is the label)
X_train = X[1:7]
Y_train = Y[1:7]
 
# Create linear regression object and predict the consumption at 9 PN (21)
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
predicted_value=regr.predict(21)

#Initialie a plot for plotting the model 
plt.scatter(X_train, Y_train,  color='black')
plt.ylabel("Kgs")
plt.xlabel("Time")
plt.ylim([-20,30])
plt.xlim([10,25])

# Plot outputs
plt.plot(X_train, regr.predict(X_train), color='blue')

#Set the size of the dot for prediction point
s = [100]

#Draw teh plot
plt.scatter(21,predicted_value,color='red',s=s)

#Annotate the prediction point 
plt.annotate('predicted 9PM ', xy=(21, predicted_value), xytext=(21,0),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.show()


print(predicted_value)

#Calculate the 50% capacity of the container
container_capacity_50pc=(container_capacity*50)/100
container_capacity_5pc=(container_capacity*5)/100

if(predicted_value<0) or (predicted_value<container_capacity_5pc):
    print("Order before 5 PM")
    generateRecommendation()
elif (predicted_value>container_capacity_5pc) and (predicted_value<container_capacity_50pc):
    print("Order after 5 PM")
    generateRecommendation()
else:
    print("Mozzarella Cheese is well stocked for today")