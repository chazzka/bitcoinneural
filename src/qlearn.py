import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

from datetime import datetime

def getFinanceDict(comodity: str, start: str, end: str):
    return pdr.get_data_yahoo(comodity, start=start, end=end).reset_index().to_dict()


def plotPrice(x,y):
    plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Close price [USD]")
    plt.show()


def dateTimeToMilliseconds(datetimeString):
    dt_obj = datetime.strptime(datetimeString,
                               '%Y-%m-%d %H:%M:%S.%f')
    millisec = dt_obj.timestamp() * 1000
    return millisec


def getLastXPercent(data, percentage):
    return data[-int(len(data)*percentage):]

def getFirstXPercent(data, percentage):
    return data[:int(len(data)*percentage)]


if __name__ == "__main__":

    yf.pdr_override()

    financedict = getFinanceDict("BTC-USD", start="2015-01-01", end="2017-01-01")


    onlyDateClose = {'Date': list(map(lambda time: time.timestamp(), financedict['Date'].values())), 'Close': list(financedict['Close'].values())}

    # sinus data
    # xdata = list(np.linspace(-np.pi, 10*np.pi, 10000))
    # onlyDateClose = {'Date': xdata, 'Close': list(map(lambda x: np.sin(x), xdata))}

    firstEighty = {'Date': getFirstXPercent(onlyDateClose['Date'], 0.8), 'Close': getFirstXPercent(onlyDateClose['Close'], 0.8)}
    lastTwenty = {'Date': getLastXPercent(onlyDateClose['Date'], 0.2), 'Close': getLastXPercent(onlyDateClose['Close'], 0.2)}

    #plotPrice(lastTwenty['Date'], lastTwenty['Close'])

    X_train =  np.asarray(firstEighty['Date']).reshape(-1,1)
    X_standard = preprocessing.StandardScaler().fit(X_train).transform(X_train).flatten()

    

    y_train = firsy_train = np.asarray(firstEighty['Close'])

    f = lambda x: [[x_] for x_ in x] # possible also reshape

    regr = MLPRegressor(hidden_layer_sizes= tuple([100]*50),max_iter=1000).fit(f(X_standard), y_train)
    predicted = regr.predict(f(X_standard))

    plt.plot(X_standard, predicted, c='b',  linewidth=5)
    plt.plot(X_standard, y_train, c='r',linewidth=2)
    plt.show()
