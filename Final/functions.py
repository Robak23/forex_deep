import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

def normalize(serie):
    mean = serie.mean()
    std = serie.std()
    z = (serie-mean)/std
    return z

def scale(training, test):
    return np.tanh(training), np.tanh(test)


def R_squared (y_pred, y_real):
    ss_res = np.sum((y_real-y_pred)**2)
    ss_tot = np.sum((y_real-np.mean(y_real))**2)
    return 1 - (ss_res/ss_tot)

def unscale(serie):
    return np.arctanh(serie)

def unnormalize(serie, mean, std):
    return (serie-mean)/std

def adf_test(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary ")