"""

@author: Jinal Shah

This script is for the webpage
front-end for this project.

The front-end will be simple with the following 
feature:

A text box for a user to enter text to be fed into 
the model. 

"""
# Importing libraries
import re
import streamlit as st 
import pandas as pd
from preprocessing import Preprocessing
import pickle
import sys

# Inserting the path
sys.path.append('../')

# A function to get the word count
def get_word_count(text:str) -> int:
    """
    get_word_count

    A function to get the word count of some text.

    inputs:
    - text: a string that indicates you want to get the word count for.

    outputs:
    - an integer representing the word count
    """
    return len(re.findall(r'[a-zA-Z_]+',text))

numerical = ['word_count','stop_word_count','stop_word_ratio','unique_word_count','unique_word_ratio',
             'count_question','count_exclamation','count_semi','count_colon','grammar_errors']

st.markdown("<h1 style='text-align: center; color: teal;'>Authentic.AI</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: teal;'>Using machine learning to detect A.I generated essays 🤖</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: teal;'>How It Works</h3>", unsafe_allow_html=True)
st.write('Paste your essay into the below text box and click submit. Within 3 minutes, you will see a probability of how likely your essay is generated by A.I')

# Creating state for inputted essay
if 'essay' not in st.session_state.keys():
    st.session_state['essay'] = ''

# Creating a form to drop the text into 
form = st.form(key='my_form')

# Need to make sure the \n are recognized as escape characters
st.session_state['text'] = form.text_input(label='Enter the Essay:').replace(r'\n','\n')
submit_button = form.form_submit_button('Submit')


# If button clicked
if submit_button:
    # Converting text to a dataframe
    input_dict = {'essay':st.session_state['text'],'word_count':get_word_count(st.session_state['text'])}
    input_df = pd.DataFrame.from_dict(input_dict,orient='index').T
    # Sending text through the preprocessing pipeline
    preprocessor = Preprocessing(input_df)
    preprocessed_input = preprocessor.preprocessing()

    # Dropping the essay column 
    preprocessed_input.drop(['essay'],axis=1,inplace=True)

    # Scaling it
    with open('../scaler.pkl','rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    preprocessed_input[numerical] = scaler.transform(preprocessed_input[numerical])
    
    # Sending into model
    with open('../models/d-tree-max-depth-3-max-features-3.pkl','rb') as model_file:
        model = pickle.load(model_file)

    # Making predictions
    probabilities = model.predict_proba(preprocessed_input.values)[:,1]
    st.write(f'There is a {round(probabilities[0] * 100,2)}% chance this essay was written by a LLM')