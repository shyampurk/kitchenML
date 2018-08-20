# Using Machine Learning to Predict Kitchen Inventory Supplies

This repository contains the accompanying source code for [this](http://radiostud.io/applying-ai-food-industry-to-streamline-process-workflow/) blog post.

Read the post carefully to understand the story and the scenario for applying predictions for a restaurant's kitchen supplies.

The programs included in this repo are used to simulate and test the supply workflow of the restaurant. 

## Prerequisites

This project uses the Python data science packages, mainly the scikit learn. All code is tested with the packages installed as part of Anaconda 3.

## Datasets

There are two data sets used in the program (all in CSV format).

1. Notification Data : Hourly weight data of stock consumption for a day. Based on the three workflow scenarios, there are three files, prefixed with "noti_"  with suffix as p, n and c denoting constant, negative and positive. These represent the three workflow scenarios. 

2. Recommendation Data : Historical sales data based on which prediction is carried out. There are two variants of this data, recom_train.csv and recom_train_doubled.csv. The second file has all the sales numbers doubled. You can use either of them. 

## Programs

1. KitchenML.py : This is the main program file which simulates the stock ordering workflow during the business hours of the restaurant.

2. KitchenMLTest.py : This program is used to generate the error and accuracy metrics for the various models applied for prediction.

3. KitchenMLTestAIC.py : Shortened version of the KitchenMLTest.py to show the min and max metrics for verifying the model. 

## Description

KitchenML.py takes in the input notification data in the form of hourly weight consumption of Mozzarella cheese and decides the ordering time along with the recommendation for order quantity.

The notification for order is calculated based on a simple linear regression model.

For predicting the recommended order quantity, the ARIMA model is used as follows.

1. The historical sales data is split into 90% training and 10% test set. 

2. For every entry in test data, the ARIMA model is initialized with the training set plus the test sample and a prediction is made.

3. The actual observation from the test dataset will be added to the training dataset for the next iteration.

4. The predictions made during the step 2 will be evaluated and an RMSE score is reported.

5. Final prediction is made by including the entire historical sales data.

Note :- The steps 3  and 4 are used for generating the MSE, RMSE for measuring the model's performance. Based on testing, this model gained 85-95% accuracy with 10-15% error.

## Running the program

KitchenML.py : This program accepts one argument as the scenario number, either 1, 2 or 3.

KitchenMLTest.py : No arguments. Note that this program will take a very long time to run as it will execute the ARIMA model with all permutations of the parameters to find the best performing model based on AIC

KitchenMLTestAIC.py : No arguments. Based on the prior observation obtained from executing KitchenMLTest.py, this program executes the best and worst performing model variant and displays the AIC metrics.

All programs have been tested be executing through the scipy IDE under Anaconda 3 on Windows.

