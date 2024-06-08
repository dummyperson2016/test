# https://chatgpt.com/share/63bcaa0e-b26d-45d6-ab70-51a936b9f176

import streamlit as st 
import joblib


ss = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title('Fraud Detection App')

st.write('''
	## Credit Card Fraud Detection
	Enter the features of the transaction to predict if it's fraudulent or not.
	''')

##
import pandas as pd 
data = pd.read_csv('creditcard_subset.csv')
features = data.columns[1:-1]
##

input_data = []
for feature in features:
	value = st.text_input(f'Enter {feature}', 0)
	input_data.append(value)

import numpy as np
input_data = np.array(input_data).reshape(1, -1)

input_data = ss.transform(input_data)

if st.button('Predict'):
	prediction = model.predict(input_data)
	if prediction[0] == 0:
		st.write("The transaction is NOT fraudulent.")
	else:
		st.write("The transaction is FRAUDULENT.")

