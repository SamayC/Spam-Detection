import cv2
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import asarray
from sklearn.model_selection import train_test_split
import streamlit as st
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

st.title("Spam Detection using Python")

with open("style.css") as stylefile:
    st.markdown(f"<style>{stylefile.read()}</style>", unsafe_allow_html=True)
    st.markdown("""<meta name="keywords" content="Neural, Network, Detection, Neural Network, Analysis, Kidney Stone Detection, Kidney Stone">
                   <meta name="description" content="Neural Network Analysis of Kidney Stone Detection">
                   <meta name="author" content="Shashank K">
                   <meta name="viewport" content="width=device-width, initial-scale=1.0">
                   <link rel="preconnect" href="https://fonts.googleapis.com">
                   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                   <link href="https://fonts.googleapis.com/css2?family=Alata&display=swap" rel="stylesheet">
                   <link href="https://fonts.googleapis.com/css2?family=Josefin+Sans:wght@200&display=swap" rel="stylesheet">""",
                unsafe_allow_html=True)
    
########################################################################################################################
try:
    st.markdown("""Spam detection using Python involves machine learning techniques to classify emails or messages as spam or non-spam based on patterns.
                Please Input the Spam Text""")

    user_input = st.text_input("Enter some text: ")

    if st.button("Predict"):
        if not user_input:
            st.warning("Please Enter the text")
        else:
            st.write(f'You entered: {user_input}')

            data=pd.read_csv(r'C:\Users\Library\PycharmProjects\spamdetection\email.csv')
            df=pd.DataFrame(data)
            # st.write(df)

            x=df.iloc[:,1].values
            y=df.iloc[:,0].values

            df.isnull().sum()
            le=LabelEncoder()
            y=le.fit_transform(y)

            cv=CountVectorizer(max_features=5000)
            x=cv.fit_transform(x).toarray()

            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

            cls=MultinomialNB()
            cls.fit(x_train,y_train)

            y_hat=cls.predict(x_test)

            confusion_matrix(y_test,y_hat)
            accuracy_score(y_test,y_hat)

            email=[]
            n=f'{user_input}'
            email.append(n)

            result = cls.predict(cv.transform(email))

            if result[0] == 0:
                st.write("its Not Spam email")
            else:
                st.write("its Spam email")
except Exception as ee:
    pass

########################################################################################################################
if st.button("Exit"):
    st.success("Thank you")
    time.sleep(2)
    st.markdown("""
        <meta http-equiv="refresh" content="0; url='https://www.google.com'" />
        """, unsafe_allow_html=True)