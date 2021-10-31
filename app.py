import streamlit as st

import numpy as np
import json

import tensorflow as tf
import tensorflow_text as text

import spacy
from spacy.lang.en import English
from utils import spacy_function, make_predictions, example_input

@st.cache()
def model_prediction(abstract):
    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''

    pred, lines = make_predictions(abstract)

    for i, line in enumerate(lines):
        if pred[i] == 'OBJECTIVE':
            objective = objective + line
        
        elif pred[i] == 'BACKGROUND':
            background = background + line
        
        elif pred[i] == 'METHODS':
            method = method + line
        
        elif pred[i] == 'RESULTS':
            result = result + line
        
        elif pred[i] == 'CONCLUSIONS':
            conclusion = conclusion + line

    return objective, background, method, conclusion, result



def main():
    st.set_page_config(
        page_title="SkimLit",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('SkimLit📄🔥')
    st.caption('An NLP model to classify abstract sentences into the role they play (e.g. objective, methods, results, etc..) to enable researchers to skim through the literature and dive deeper when necessary.')

    col1, col2 = st.columns(2)

    with col1:
        st.write('#### Entre Abstract Here !!')
        abstract = st.text_area(label='', height=50)
        # model = st.selectbox('Choose Model', ('Simple Model -> 82%', "Beart Model -> 89%"))

        agree = st.checkbox('Show Example Abstract')
        if agree:
            st.info(example_input)

        predict = st.button('Extract !')
    
    # make prediction button logic
    if predict:
        with st.spinner('Wait for prediction....'):
            objective, background, methods, conclusion, result = model_prediction(abstract)
        with col2:
            st.markdown(f'### Objective : ')
            st.write(f'{objective}')
            st.markdown(f'### Background : ')
            st.write(f'{background}')
            st.markdown(f'### Methods : ')
            st.write(f'{methods}')
            st.markdown(f'### Result : ')
            st.write(f'{result}')
            st.markdown(f'### Conclusion : ')
            st.write(f'{conclusion}')



if __name__=='__main__': 
    main()