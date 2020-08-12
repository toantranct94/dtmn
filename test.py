# Numerical libraries
import numpy as np   

# Import Polynomial Regression library
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

# Linear Regression Library
from sklearn.linear_model import LinearRegression

# Cross-Validation related Libraries
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#Import Lasso for Dimension Reduction
from sklearn.linear_model import Lasso

# For splitting the dataset into test and Train
from sklearn.model_selection import train_test_split

# For K-Means Clustering
from sklearn.cluster import KMeans

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import metrics

# to handle data in form of rows and columns 
import pandas as pd    

# importing ploting libraries
import matplotlib.pyplot as plt 

from scipy.stats import zscore

powerPlant = pd.read_csv("Folds5x2_pp.csv", header=0 , names = ['Temp','Vacuum','Pressure','Humidity','Power'])

print(powerPlant.shape)

powerPlant.head()

powerPlant.describe().T

# print(powerPlant.describe().T)

power_z = powerPlant.apply(zscore)

print(power_z.shape)

# print(power_z.head())

# print(power_z.describe().T)

x = powerPlant[['Temp', 'Vacuum', 'Pressure', 'Humidity']]
y = powerPlant.Power

x_intermediate, x_test, y_intermediate, y_test = train_test_split(x,y,  test_size = 0.2, random_state =1, shuffle = True)

# train/validation split (gives us train and validation sets)
x_train, x_val, y_train, y_val = train_test_split(x_intermediate, y_intermediate,  test_size = 0.25, random_state =100, shuffle = True)

# Various Degrees at which we will test the Polynomial Regression Model
degrees = [1,2,3,4,5,6]
val_error = []
regression_model = LinearRegression()

scaler = PolynomialFeatures(degree = 4, interaction_only= True)
data_scale = scaler.fit_transform(powerPlant)
x_scale = data_scale[:,:4]
y_scale = data_scale[:,4:5]

x_train, x_test, y_train, y_test = train_test_split(x_scale, y,  test_size = 0.25, random_state =100, shuffle = True)


reg = regression_model.fit(x_train, y_train)

predicted_power = reg.predict(x_test)

print(predicted_power)

## TESTING

# y_test = scaler.inverse_transform(y_test)

# predicted_power = scaler.inverse_transform(predicted_power)

mse = mean_squared_error(y_test, predicted_power)

mae = mean_absolute_error(y_test, predicted_power)

print("MSE: {}".format(mse))

print("MAE: {}".format(mae))

### Cross Validation for Interaction Only
# for degree in degrees:
#     poly = PolynomialFeatures( degree = degree, interaction_only= True)
#     x_poly = poly.fit_transform(x_intermediate)
#     reg = regression_model.fit(x_poly,y_intermediate)
#     errors = np.sum( -cross_val_score(reg, x_poly, y_intermediate ,scoring= 'neg_mean_absolute_error', cv=10))
    
#     val_error.append(np.sqrt(errors))

    # RMSE
# print(val_error)
# print('Lowest Error rate for the degree: {}'.format(degrees[np.argmin(val_error)]))