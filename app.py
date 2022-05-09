# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:29:25 2022

@author: Souleymane
"""

from flask import Flask
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import pickle, dill
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import bz2file as bz2

app = Flask(__name__)

def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data

client_list = pickle.load(open('customer_list.pickle', 'rb'))
model = pickle.load(open('model.pickle', 'rb'))
df = decompress_pickle("test_df.pbz2")
imputer = pickle.load(open('impute.pickle', 'rb'))
scaler = pickle.load(open('scale.pickle', 'rb'))
clients_all = pickle.load(open('clients.pickle','rb'))

df = df.reset_index(drop = True)
clients_all = pd.DataFrame(clients_all)
clients_all = clients_all.reset_index(drop=True)

@app.route('/login', defaults = {'cust_id' : ''})
@app.route('/login/<cust_id>', methods = ["GET"])

def login(cust_id):
    if cust_id == '':
        return_value = 'Welcome. Please select a client ID.'
    else:
        return_value = cust_id        
    return return_value

@app.route('/login/predict/<cust_id>', methods = ["GET"])

def predict(cust_id):
    
    print("custid :" + str(cust_id))
    cust_index = clients_all[clients_all.SK_ID_CURR == int(cust_id)].index[0]
    cust_info = pd.DataFrame(df.iloc[cust_index,:]).T
    print(cust_info)
    cust_info = imputer.transform(cust_info)
    cust_info = scaler.transform(cust_info)
    proba = model.predict_proba(cust_info)[:,1]
    proba = proba[0]
    proba = round(proba, 2)

    return proba.astype(str)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 443)

