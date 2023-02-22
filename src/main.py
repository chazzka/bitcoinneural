import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

from datetime import datetime

import itertools as it

from collections import namedtuple

import sys
import pprint


def getFinanceDict(comodity: str, start: str, end: str):
    return pdr.get_data_yahoo(comodity, start=start, end=end).reset_index().to_dict()


def plotPrice(x, y):
    plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Close price [USD]")
    plt.show()


def dateTimeToMilliseconds(datetimeString):
    dt_obj = datetime.strptime(datetimeString,
                               '%Y-%m-%d %H:%M:%S.%f')
    millisec = dt_obj.timestamp() * 1000
    return millisec


def getLastXPercent(data, percentage, a=0):
    return data[-int(len(data)*percentage) + a:]


def getFirstXPercent(data, percentage, a=0):
    return data[:int(len(data)*percentage) + a]


def getLastWithPercent(data, percentage, a=0):
    return data[int(len(data)*percentage)-1 + a]


def fitAndPredictPartial(model, X, y):
    return model.partial_fit(X[:-1].reshape(-1, 1), y).predict(X[-1].reshape(-1, 1))[0]


class Profit:
    def __init__(self, bitcoin, dollars):
        self.bitcoin = bitcoin
        self.dollars = dollars
    def __repr__(self):
        return f"bitcon - {self.bitcoin} doll- {self.dollars}"
    


class BuyClose:
    def __init__(self, shouldBuy, close):
        self.shouldBuy = shouldBuy
        self.close = close


def performAction(profit: Profit, buyClose: BuyClose):
    return Profit(profit.bitcoin + profit.dollars / buyClose.close, 0) if buyClose.shouldBuy else Profit(0, profit.dollars + profit.bitcoin * buyClose.close)



if __name__ == "__main__":

    #The results of the models are achieved on the Bitcoin dataset
    # taken from Yahoo! Finance from January 1, 2015, to September 23, 2021 (a total of 2458 days). The top 80% of the data
    # is applied to train for predicting closing prices by LSTM.
    # The last 20% of data will be applied to validate the models

    yf.pdr_override()

    financedict = getFinanceDict(
        "BTC-USD", start="2015-01-01", end="2021-09-24")

    onlyDateClose = {'Date': list(map(lambda time: time.timestamp(), financedict['Date'].values(
    ))), 'Close': list(financedict['Close'].values()), 'Open': list(financedict['Open'].values())}

    print(len(onlyDateClose['Date']))
    
    X = np.asarray(onlyDateClose['Date'])
    Y = np.asarray(onlyDateClose['Close'])
    # X_standard = preprocessing.StandardScaler().fit(X).transform(X).flatten()
    X_standard = ((X-X.mean()) / X.std(ddof=0))

    firstEighty = {'Date': getFirstXPercent(X_standard, 0.80), 'Close': getFirstXPercent(
        onlyDateClose['Close'], 0.80), 'Open': getFirstXPercent(onlyDateClose['Open'], 0.80)}

    lastTwenty = {'Date': getLastXPercent(X_standard, 0.20), 'Close': getLastXPercent(
        onlyDateClose['Close'], 0.20), 'Open': getLastXPercent(onlyDateClose['Open'], 0.20)}

    plotPrice(lastTwenty['Date'], lastTwenty['Close'])

    def f(x): return [[x_] for x_ in x]  # possible also reshape

    x_train = len(firstEighty['Date'])
    y_train = np.asarray(firstEighty['Close'])

    model = MLPRegressor(hidden_layer_sizes=tuple(
        [100]*50), max_iter=100, solver='adam')

    # -1 = vem jeden předchozí den
    # TODO: -7 předchozí týden
    predictedV = list(map(lambda i: fitAndPredictPartial(
        model, X_standard[i-1:i+1], Y[i-1:i]), range(10, len(X_standard))))

    # spočítej reward
    reward = list(map(lambda actualClose,
                  predictedClose: predictedClose - actualClose, Y[10:], predictedV))

    # plot reward
    plt.plot(X_standard[10:], Y[10:], label='Close')
    plt.plot(X_standard[10:], predictedV, label='predicted Close')
    plt.legend()
    plt.show()

    # do it only on last 20 percent
    openPrice = lastTwenty['Open']
    predictedPrice = predictedV[-len(openPrice):]
    closePrice = lastTwenty['Close']

    # udělej funnkci co bude provádět nakup na burze podle .. open price vs predicted close
    # buy - true
    # sell - false
    shouldBuy = [x > y for x, y in zip(predictedPrice, openPrice)]

    # ted vis kdy mas nakupovat a kdy drzet, ale nakupujeme za close price
    buyCloseArray = [BuyClose(x, y) for x, y in zip(shouldBuy, closePrice)]

    profits = list(it.accumulate(buyCloseArray, func=performAction, initial=Profit(0, 300)))

    pp = pprint.PrettyPrinter()

    pp.pprint(profits)

    print(len(openPrice))
    print(len(profits))

    # now máš pole částek, někdy bitcoin, někdy dolar (nakonci snad dolar) - TODO:
    # profits[1:] without first 300
    dolarValues = list([profit.dollars + profit.bitcoin * close for profit, close in zip(profits[1:], closePrice)])

    def getDollars(previousDollars, profit):
        return previousDollars if profit.dollars == 0 else profit.dollars

    # function to return always last dolar value (do not convert everytime from bitcion)
    onlyDolar = list(it.accumulate(profits[1:], func=getDollars, initial=300))

    #plt.plot(range(0,len(lastTwenty['Date'])), dolarValues)
    plt.plot(range(0,len(lastTwenty['Date'])), onlyDolar[1:])
    plt.show()

    # když bude open 300 a predicted uzvaírka 400 je výsledek kladný (predicted-open) je dobré nakoupit (za vše)
    # když bude open 300 a predicted uzvaírka 200 je výsledek záporný (predicted-open) je dobré prodat (za vše)
    # když nemáme peníze na nákup nebo nemáme co prodat tak spadne do hold
