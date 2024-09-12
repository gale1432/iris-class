#Universidad Autónoma de Chihuahua
#Machine Learning
#Proyecto Iris Clasificacion
#Gael Aristides Hinojos Ramírez

import pandas as pd
import numpy as np
import streamlit as st
import joblib

def predict(data):
    log_reg = joblib.load('logistic_regression.sav')
    pipeline = joblib.load('pipeline.sav')
    pipelined_data = pipeline.fit_transform(data)
    predicted_class = log_reg.predict(pipelined_data)
    #if predicted_class[0] == 0:
    #    return 'Iris-setosa'
    #elif predicted_class[0] == 1:
    #    return 'Iris-versicolor'
    #elif predicted_class[0] == 2:
    #    return 'Iris-virginica'
    return predicted_class[0]
    
st.title("Clasificar tipos de iris")
st.header("Variables")
col1, col2 = st.columns(2)

with col1:
    sep_len_cm = st.number_input(label="Sepal Length (in cm)")
    sep_wid_cm = st.number_input(label="Sepal Width (in cm)")
with col2:
    pet_len_cm = st.number_input(label="Petal Length (in cm)")
    pet_wid_cm = st.number_input(label="Petal Width (in cm)")

if st.button("Classify iris"):
    data = {'slc': sep_len_cm, 'swc': sep_wid_cm, 'plc': pet_len_cm, 'pwc': pet_wid_cm}
    df = pd.DataFrame([list(data.values())], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    result = predict(df)
    st.text("La flor es una: {}".format(result))