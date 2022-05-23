import streamlit as st
import pandas as pd
import numpy as np
import adtk
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from adtk.detector import PersistAD
from adtk.detector import InterQuartileRangeAD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from adtk.visualization import plot
st.title("Anomaly Detection on Cryptocurrency Prices")
st.write("This particular app has been made in order to detect anomalies on Crypto values and provide to the user, the ability to choose and optimize different algorithms.")
st.markdown('**In order for the application to function properly, please follow the instructions below:**')
st.write("1. Select the desired Cryptocurrency Coin")
st.write("2. Select the value on which you want to perform Anomaly detection.")
st.write("3. Select the desired detector on <Select Anomaly Detector> sidebar option.")
st.write("4. Select the desired parameters of the model on the sidebar.")

crypto_name = st.sidebar.selectbox("Select Coin", ("Bitcoin", "Ethereum", "Binance", "Tether", "USD Coin"))
value_name = st.sidebar.selectbox("Select Values", ("Open Price", "Volume", "Change(%)", "High", "Low"))

detector_name = st.sidebar.selectbox("Select Anomaly Detector", ("PersistAD", "InterQuartileRangeAD", "IsolationForest", "LocalOutlierFactor"))
st.write(detector_name)
st.set_option('deprecation.showPyplotGlobalUse', False)

def add_parameters(detector_name):
    params = dict()
    if detector_name == "PersistAD":
        st.info('This detector compares time series values with the values of their preceding time windows, and identifies a time point as anomalous if the change of value from its preceding average or median is anomalously large.')
        c = st.sidebar.number_input("c", min_value=0.00,  max_value=100.00, step=1e-2)
        st.sidebar.info(' Factor used to determine the bound of normal range based on historical interquartile range. The larger the value of c is, the bigger is the range where within it, a point is considered as an inlier.')
        window = st.sidebar.slider("window", 1, 100)
        st.sidebar.info('Size of the preceding time window. For example, if set to 4, it compares with previous four time points.')
        params['c']= c
        params['window']= window

    elif detector_name == "IsolationForest":
        st.info('Isolation Forest recursively generates partitions on the dataset by randomly selecting a feature and then randomly selecting a split value for the feature. Presumably the anomalies need fewer random partitions to be isolated compared to "normal" points in the dataset, so the anomalies will be the points which have a smaller path length in the tree, path length being the number of edges traversed from the root node.')
        contamination = st.sidebar.number_input("contamination rate", min_value=0.000,  max_value= 0.500, step=1e-3)
        st.sidebar.info('The proportion of outliers in the data set.')
        n_estimators = st.sidebar.slider('n_estimators', 1, 500)
        st.sidebar.info('The number of decision trees to be created in a random forest.')
        params['contamination']= contamination
        params['n_estimators']= n_estimators
    elif detector_name == "LocalOutlierFactor":
        st.info('LocalOutlierFactor measures (k) neighbors of a specific point in order to find the density of each point and compare it with the density of other points. If (k) is a small value it compares the densities within a small area, if large it compares the densities within a larger area.')
        contamination = st.sidebar.number_input("contamination rate", min_value=0.000,  max_value= 0.500, step=1e-3)
        st.sidebar.info('The proportion of outliers in the data set.')
        n_neighbors = st.sidebar.number_input("n neighbors", min_value=0,  max_value= 100, step = 1)
        st.sidebar.info('Number of neighbors to use (k) . A large value of neighbors might miss some outliers, while a small number of neighbors might detect some inliers as outliers.')
        params['contamination']= contamination
        params['n_neighbors']= n_neighbors
    else:
        st.info('This detector compares time series values with 1st and 3rd quartiles of historical data, and identifies time points as anomalous when differences are beyond the inter-quartile range (IQR) times a user-given factor c.')
        c = st.sidebar.number_input("c", min_value=0.00,  max_value=100.00, step=1e-2)
        st.sidebar.info('Factor used to determine the bound of normal range (betweeen Q1-c*IQR and Q3+c*IQR).')
        params['c']= c
    
    return params
params = add_parameters(detector_name)

def get_detector(detector_name, params):
    if detector_name == "PersistAD":
        detector = PersistAD(c = params['c'], window = params['window'])
    elif detector_name == "IsolationForest":
        detector = IsolationForest(contamination = params['contamination'], n_estimators = params['n_estimators'])
    elif detector_name == "LocalOutlierFactor":
        detector = LocalOutlierFactor(contamination = params['contamination'], n_neighbors = params['n_neighbors'], novelty=False)
    else:
        detector = InterQuartileRangeAD(c = params['c'])
    return detector
detector = get_detector(detector_name, params)

if crypto_name == 'Bitcoin':
    url = 'https://gadgets360.com/finance/bitcoin-price-history'
elif crypto_name == 'Ethereum':
    url = 'https://gadgets360.com/finance/ethereum-price-history'
elif crypto_name == 'Binance':
    url = 'https://gadgets360.com/finance/binance-coin-price-history'  
elif crypto_name == 'Tether':
    url = 'https://gadgets360.com/finance/tether-price-history' 
else:
    url = 'https://gadgets360.com/finance/usd-coin-price-history'   

r = requests.get(url)
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

datelist = []
dates = soup.find_all('div', class_ = '_flx _cpnm')

for i in dates:
  text = i.text
  datelist.append(text)

open = soup.find_all('td', class_ = '_lft')
open = open[0::5]

openlist = []

for i in open:
  opens = i.text
  openlist.append(opens)

high = soup.find_all('td', class_ = '_lft')
high = high[1::5]

highlist = []

for i in high:
  highs = i.text
  highlist.append(highs)

low = soup.find_all('td', class_ = '_lft')
low = low[2::5]

lowlist = []

for i in low:
  lows = i.text
  lowlist.append(lows)

volume = soup.find_all('td', class_ = '_lft')
volume = volume[4::5]

volumelist = []

for i in volume:
  volumes = i.text
  volumelist.append(volumes)

changelist = []
changes = soup.find_all('td', class_ = '_chngt')

for i in changes:
  change = i.text
  changelist.append(change)

df = pd.DataFrame(
    {'Date': datelist[0:50],
     'Open_Price': openlist[0:50],
     'High_Price': highlist[0:50],
     'Low_Price': lowlist[0:50],
     'Volume': volumelist[0:50],
     'Change': changelist[0:50]
    })


df["Open_Price"] = df["Open_Price"].str.replace(r'\D', '')
df["Open_Price"] = pd.to_numeric(df["Open_Price"])
df["Open_Price"] = df["Open_Price"] * 0.013

df["High_Price"] = df["High_Price"].str.replace(r'\D', '')
df["High_Price"] = pd.to_numeric(df["High_Price"])
df["High_Price"] = df["High_Price"] * 0.013

df["Low_Price"] = df["Low_Price"].str.replace(r'\D', '')
df["Low_Price"] = pd.to_numeric(df["Low_Price"])
df["Low_Price"] = df["Low_Price"] * 0.013

df["Volume"] = df["Volume"].str.replace(r'\D', '')
df["Volume"] = pd.to_numeric(df["Volume"])
df["Volume"] = df["Volume"] * 0.013

df['Change'] = df['Change'].str.replace("%","")
df['Change'] = df['Change'].str.replace("+","")
df["Change"] = pd.to_numeric(df["Change"])


df = df.set_index('Date')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

st.write(df)

if value_name == 'Open Price':

    if detector_name == "InterQuartileRangeAD":
        anomalies = detector.fit_predict(df[['Open_Price']])
        anomalies_true = anomalies.loc[anomalies['Open_Price'] == True, ['Open_Price']]
        a = df[df.index.isin(anomalies_true.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Open_Price'])
        plot(df[['Open_Price']], anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'PersistAD':
        anomalies_A = detector.fit_predict(df[['Open_Price']])
        anomalies_true_A = anomalies_A.loc[anomalies_A['Open_Price'] == 1, ['Open_Price']]
        a = df[df.index.isin(anomalies_true_A.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Open_Price'])
        plot(df[['Open_Price']], anomaly=anomalies_A, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'IsolationForest':
        df['anomalies'] = detector.fit_predict(df[['Open_Price']])
        a = df.loc[df['anomalies'] == -1, ['Open_Price']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Open_Price'], color='blue', label = 'Normal Open_Price')
        ax.scatter(a.index, a['Open_Price'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

    else:
        df['anomalies'] = detector.fit_predict(df[['Open_Price']])
        a = df.loc[df['anomalies'] == -1, ['Open_Price']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Open_Price'], color='blue', label = 'Normal Open_Price')
        ax.scatter(a.index, a['Open_Price'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

elif value_name == 'Volume':

    if detector_name == "InterQuartileRangeAD":
        anomalies = detector.fit_predict(df[['Volume']])
        anomalies_true = anomalies.loc[anomalies['Volume'] == True, ['Volume']]
        a = df[df.index.isin(anomalies_true.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Volume'])
        plot(df[['Volume']], anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'PersistAD':
        anomalies_A = detector.fit_predict(df[['Volume']])
        anomalies_true_A = anomalies_A.loc[anomalies_A['Volume'] == 1, ['Volume']]
        a = df[df.index.isin(anomalies_true_A.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Volume'])
        plot(df[['Volume']], anomaly=anomalies_A, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'IsolationForest':
        df['anomalies'] = detector.fit_predict(df[['Volume']])
        a = df.loc[df['anomalies'] == -1, ['Volume']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Volume'], color='blue', label = 'Normal Volume')
        ax.scatter(a.index, a['Volume'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

    else:
        df['anomalies'] = detector.fit_predict(df[['Volume']])
        a = df.loc[df['anomalies'] == -1, ['Volume']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Volume'], color='blue', label = 'Normal Volume')
        ax.scatter(a.index, a['Volume'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

elif value_name == 'Change(%)':

    if detector_name == "InterQuartileRangeAD":
        anomalies = detector.fit_predict(df[['Change']])
        anomalies_true = anomalies.loc[anomalies['Change'] == True, ['Change']]
        a = df[df.index.isin(anomalies_true.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Change'])
        plot(df[['Change']], anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'PersistAD':
        anomalies_A = detector.fit_predict(df[['Change']])
        anomalies_true_A = anomalies_A.loc[anomalies_A['Change'] == 1, ['Change']]
        a = df[df.index.isin(anomalies_true_A.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Change'])
        plot(df[['Change']], anomaly=anomalies_A, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'IsolationForest':
        df['anomalies'] = detector.fit_predict(df[['Change']])
        a = df.loc[df['anomalies'] == -1, ['Change']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Change'], color='blue', label = 'Normal Change(%)')
        ax.scatter(a.index, a['Change'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

    else:
        df['anomalies'] = detector.fit_predict(df[['Change']])
        a = df.loc[df['anomalies'] == -1, ['Change']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Change'], color='blue', label = 'Normal Change(%)')
        ax.scatter(a.index, a['Change'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

elif value_name == 'High':

    if detector_name == "InterQuartileRangeAD":
        anomalies = detector.fit_predict(df[['High_Price']])
        anomalies_true = anomalies.loc[anomalies['High_Price'] == True, ['High_Price']]
        a = df[df.index.isin(anomalies_true.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['High_Price'])
        plot(df[['High_Price']], anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()


    elif detector_name == 'PersistAD':
        anomalies_A = detector.fit_predict(df[['High_Price']])
        anomalies_true_A = anomalies_A.loc[anomalies_A['High_Price'] == 1, ['High_Price']]
        a = df[df.index.isin(anomalies_true_A.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['High_Price'])
        plot(df[['High_Price']], anomaly=anomalies_A, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'IsolationForest':
        df['anomalies'] = detector.fit_predict(df[['High_Price']])
        a = df.loc[df['anomalies'] == -1, ['High_Price']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['High_Price'], color='blue', label = 'Normal High')
        ax.scatter(a.index, a['High_Price'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

    else:
        df['anomalies'] = detector.fit_predict(df[['High_Price']])
        a = df.loc[df['anomalies'] == -1, ['High_Price']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['High_Price'], color='blue', label = 'Normal High')
        ax.scatter(a.index, a['High_Price'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

elif value_name == 'Low':

    if detector_name == "InterQuartileRangeAD":
        anomalies = detector.fit_predict(df[['Low_Price']])
        anomalies_true = anomalies.loc[anomalies['Low_Price'] == True, ['Low_Price']]
        a = df[df.index.isin(anomalies_true.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Low_Price'])
        plot(df[['Low_Price']], anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'PersistAD':
        anomalies_A = detector.fit_predict(df[['Low_Price']])
        anomalies_true_A = anomalies_A.loc[anomalies_A['Low_Price'] == 1, ['Low_Price']]
        a = df[df.index.isin(anomalies_true_A.index)]
        st.markdown('**Anomaly Indices**')
        st.write(a['Low_Price'])
        plot(df[['Low_Price']], anomaly=anomalies_A, anomaly_color="red", anomaly_tag="marker")
        st.pyplot()

    elif detector_name == 'IsolationForest':
        df['anomalies'] = detector.fit_predict(df[['Low_Price']])
        a = df.loc[df['anomalies'] == -1, ['Low_Price']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Low_Price'], color='blue', label = 'Normal Low')
        ax.scatter(a.index, a['Low_Price'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()

    else:
        df['anomalies'] = detector.fit_predict(df[['Low_Price']])
        a = df.loc[df['anomalies'] == -1, ['Low_Price']]
        st.markdown('**Anomaly Indices**')
        st.write(a)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df['Low_Price'], color='blue', label = 'Normal Low')
        ax.scatter(a.index, a['Low_Price'], color='red', label = 'Anomaly')
        plt.legend()
        plt.show()
        st.pyplot()