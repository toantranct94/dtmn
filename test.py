# Numerical libraries
import numpy as np   
# Import Polynomial Regression library
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
# Linear Regression Library
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
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

import seaborn as sns

powerPlant = pd.read_csv("Folds5x2_pp.csv", header=0 , names = ['Temp','Vacuum','Pressure','Humidity','Power'])

print(powerPlant.shape)


# print(powerPlant.describe().T)

power_z = powerPlant.apply(zscore)

print(power_z.shape)

# print(power_z.head())

# print(power_z.describe().T)

# corr = powerPlant.corr()

# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(1,200, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot  = True)

x = powerPlant[['Temp', 'Vacuum', 'Pressure', 'Humidity']]
y = powerPlant['Power']

# x_intermediate, x_test, y_intermediate, y_test = train_test_split(x,y,  test_size = 0.2, random_state =1, shuffle = True)

# # train/validation split (gives us train and validation sets)
# x_train, x_val, y_train, y_val = train_test_split(x_intermediate, y_intermediate,  test_size = 0.25, random_state =100, shuffle = True)

# Various Degrees at which we will test the Polynomial Regression Model


# regression_model = LinearRegression()

# scaler = PolynomialFeatures(degree = 4, interaction_only= True)

# # scaler = MinMaxScaler()

# scaler_y = MinMaxScaler()

# # x_scale = data_scale[:,:4]

# # y_scale = data_scale[:,4:5]

# x_scale = scaler.fit_transform(x)

# y_scale = scaler_y.fit_transform(y.values)

# x_train, x_test, y_train, y_test = train_test_split(x_scale, y,  test_size = 0.25, random_state =100, shuffle = True)

# reg = regression_model.fit(x_train, y_train)

# predicted_power = reg.predict(x_test)

# print(predicted_power)

# ## TESTING

# # y_test = scaler.inverse_transform(y_test)

# # predicted_power = scaler.inverse_transform(predicted_power)

# mse = mean_squared_error(y_test, predicted_power)

# mae = mean_absolute_error(y_test, predicted_power)

# print("MSE: {}".format(mse))

# print("MAE: {}".format(mae))


########### Defining Functions to calculate Error Metrics ###############

## Returns in-sample error for already fit model
def calc_train_error(X_train, y_train, model, scaler=None):
    predictions = model.predict(X_train)
    if scaler is not None:
        y_train = scaler.inverse_transform(y_train)
        predictions = scaler.inverse_transform(predictions)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, predictions)
    return rmse, mae
## Returns out-of-sample error for already fit model   
def calc_validation_error(X_test, y_test, model, scaler=None):
    predictions = model.predict(X_test)
    if scaler is not None:
        y_test = scaler.inverse_transform(y_test)
        predictions = scaler.inverse_transform(predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    return rmse, mae

## fits model and returns the RMSE for in-sample error and out-of-sample error   
def calc_metrics(X_train, y_train, X_test, y_test, model, scaler=None):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    # 
    train_error = calc_train_error(X_train, y_train, model, scaler)
    validation_error = calc_validation_error(X_test, y_test, model, scaler)
    return train_error, validation_error

def run_kfold_poly(k, lr_or_rf=True):
    degrees = [1,2,3,4,5,6]
    kf = KFold(n_splits=k, shuffle = True)
    x_array = np.asarray(x)
    result = []
    for degree in degrees:
        total_train_rmse = []
        total_train_mae = []
        total_val_rmse = []
        total_val_mae = []
        
        # Instantiating Model
        for train_index, val_index in kf.split(x_array,y):
            if lr_or_rf:
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=500, max_depth=5)
            poly = PolynomialFeatures(degree = degree, interaction_only= True)
            x_poly = poly.fit_transform(x_array)
            # Split data
            x_train, x_val = x_poly[train_index], x_poly[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model.fit(x_train, y_train)
            # Calculate Error
            train_error, val_error = calc_metrics(x_train, y_train, x_val, y_val, model)

            train_rmse = train_error[0]
            train_mae = train_error[1]
            val_rmse = val_error[0]
            val_mae = val_error[1]

            total_train_rmse.append(train_rmse)
            total_val_rmse.append(val_rmse)
            total_train_mae.append(train_mae)
            total_val_mae.append(val_mae)
        if lr_or_rf:
            print("Linear Regression")
        else:
            print("Random Forest Regression")
        print("DEGREE: {}".format(degree))
        print("TRAINING RMSE: {} ------ VAL RMSE {}".format(round(np.mean(total_train_rmse), 4), round(np.mean(total_val_rmse), 4)))
        print("TRAINING MAE: {} ------ VAL MAE {}".format(round(np.mean(total_train_mae), 4), round(np.mean(total_val_mae), 4)))

def run_kfold_scale(k, lr_or_rf=True, min_max_or_standar=True):
    kf = KFold(n_splits=k, shuffle = True)
    x_array = np.asarray(x)

    total_train_rmse = []
    total_train_mae = []
    total_val_rmse = []
    total_val_mae = []

    if min_max_or_standar:
        scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        y_scaler = StandardScaler()

    x_scale = scaler.fit_transform(x_array)
    y_scale = y_scaler.fit_transform(y.values.reshape(-1,1))
    # Instantiating Model
    for train_index, val_index in kf.split(x_scale,y):
        if lr_or_rf:
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=500, max_depth=5)
        # Split data
        x_train, x_val = x_scale[train_index], x_scale[val_index]
        y_train, y_val = y_scale[train_index], y_scale[val_index]
        model.fit(x_train, y_train)
        # Calculate Error

        train_error, val_error = calc_metrics(x_train, y_train, x_val, y_val, model, y_scaler)

        train_rmse = train_error[0]
        train_mae = train_error[1]
        val_rmse = val_error[0]
        val_mae = val_error[1]

        total_train_rmse.append(train_rmse)
        total_val_rmse.append(val_rmse)
        total_train_mae.append(train_mae)
        total_val_mae.append(val_mae)

    if lr_or_rf:
        print("Linear Regression")
    else:
        print("Random Forest Regression")

    if min_max_or_standar:
        print("MinMaxScaler")
    else:
        print("StandardScaler")

    print("TRAINING RMSE: {} ------ VAL RMSE {}".format(round(np.mean(total_train_rmse), 4), round(np.mean(total_val_rmse), 4)))
    print("TRAINING MAE: {} ------ VAL MAE {}".format(round(np.mean(total_train_mae), 4), round(np.mean(total_val_mae), 4)))


if __name__ == "__main__":
    run_kfold_poly(k=10, lr_or_rf=True)
    run_kfold_poly(k=10, lr_or_rf=False)
    run_kfold_scale(k=10, r_or_rf=True, min_max_or_standar=True)
    run_kfold_scale(k=10, r_or_rf=False, min_max_or_standar=True)
    run_kfold_scale(k=10, r_or_rf=True, min_max_or_standar=False)
    run_kfold_scale(k=10, r_or_rf=False, min_max_or_standar=False)