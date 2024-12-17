
import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('final_model_xgb.pkl','rb') as file:
    model = pickle.load(file)

with open ('transformer.pkl','rb') as file:
    pt = pickle.load(file)

def prediction(input_list):
    tran_data = pt.transform([[input_list[0],input_list[3]]])
    input_list[0] = tran_data[0][0]
    input_list[3] = tran_data[0][1]
    input_list = np.array(input_list,dtype=object)
    pred = model.predict_proba([input_list])[:,1][0]

    if pred>0.5:
        return f'This Booking is more Likely to get Canceled: Chances {round(pred,2)}'
    else:
        return f'This Booking is Less Likely to get Canceled {round(pred,2)}'

def main():
    st.title('INN HOTEL GROUP')
    lt = st.text_input('Enter the lead time.')
    mst = (lambda x: 1 if x=='Online' else 0)(st.selectbox('Enter the Type of Booking',['Online','Offline']))
    spcl = st.selectbox('Select the Number of Special Request Made',[0,1,2,3,4,5])
    price = st.text_input('Enter the Price Offered for the Room')
    adults = st.selectbox('Select the No. of Adults in Booking',[0,1,2,3,4])
    wkend = st.text_input('Enter the Weekend Nights in the Booking')
    wk = st.text_input('Enter the Weeknights in Booking')
    park = (lambda x : 1 if x=='Yes' else 0)(st.selectbox('Is Parking included in the Booking',['Yes','No']))
    month = st.slider('What will be month of Arrival',min_value=1,max_value=12,step=1)
    day = st.slider('What will be day of Arrival',min_value=1,max_value=31,step=1)
    weekd_lambda = (lambda x: 0 if x == 'Mon' else 1 if x=='Tue' else 2 if x=='Wed' else 3 if x=='Thus' else 4 if x=='Fri' else
                   5 if x=='Sat' else 6)
    weekd = weekd_lambda(st.selectbox('What is the weekday of Arrival',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))

    inp_list = [lt,mst,spcl,price,adults,wkend,park,wk,month,day,weekd]

    if st.button('Predict'):
        response = prediction(inp_list)
        st.success(response)

if __name__=='__main__':
    main()
