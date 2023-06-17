import streamlit as st
import pandas as pd
import numpy as np
import pickle as pickle

st.set_page_config(layout="wide")

filename = "D:/DS/Semester_1/MachineLearning/TermProject/Churn_LIME_Predict.sav"

#df = df2[['Price', 'Age_08_04', 'KM', 'Fuel_Type', 'Automatic', 'Gears']]
loaded_model = pickle.load(open(filename, 'rb'))
                           
st.title('Telecom Churn Prediction Classification Web App')
st.write('This is a web app to predict the price of a Toytoa Car using\
        several features that you can see in the sidebar. Please adjust the\
        value of each feature. After that, click on the Predict button at the bottom to\
        see the prediction of the regressor.')
#st.image('C:/Users/Dell/Downloads/Telco churn1.jpg', caption=None, width=None, use_column_width=False,)



with st.sidebar:
    TotalCharges= st.number_input(label='Total Charges',min_value = 0.0,
                        max_value = 10000.0 ,
                        value = 10.0,
                        step = 5.0)    
      
    tenure= st.number_input(label='tenure',min_value = 0.0,
                        max_value = 70.0 ,
                        value = 5.0,
                        step = 1.0)  


features = {
  'TotalCharges':TotalCharges,
  'tenure':tenure}
  

features_df  = pd.DataFrame([features])

st.table(features_df)
col1, col2 = st.columns((1,2))

with col1:
    prButton = st.button('Predict')
with col2: 
    if prButton:    
        prediction = loaded_model.Churn(features_df)    
        st.write(' Based on feature values, the customer churn is '+ str(int(prediction)))