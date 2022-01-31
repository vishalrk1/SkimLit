import streamlit as st
import torch
import spacy
# from spacy.lang.en import English
# from utils import spacy_function, make_predictions, example_input

from Dataset import SkimlitDataset
from Embeddings import get_embeddings
from Model import SkimlitModel
from Tokenizer import Tokenizer
from LabelEncoder import LabelEncoder
from MakePredictions import make_skimlit_predictions

MODEL_PATH = 'PyTorch/utils/skimlit-model-final-1.pt'
TOKENIZER_PATH = 'PyTorch/utils/tokenizer.json'
LABEL_ENOCDER_PATH = "PyTorch/utils/label_encoder.json"
EMBEDDING_FILE_PATH = 'PyTorch/utils/glove.6B.300d.txt'

@st.cache()
def create_utils(model_path, tokenizer_path, label_encoder_path, embedding_file_path):
    tokenizer = Tokenizer.load(fp=tokenizer_path)
    label_encoder = LabelEncoder.load(fp=label_encoder_path)
    embedding_matrix = get_embeddings(embedding_file_path, tokenizer, 300)
    model = SkimlitModel(embedding_dim=300, vocab_size=len(tokenizer), hidden_dim=128, n_layers=3, linear_output=128, num_classes=len(label_encoder), pretrained_embeddings=embedding_matrix)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(model)
    return model, tokenizer, label_encoder

def model_prediction(abstract, model, tokenizer, label_encoder):
    objective = ''
    background = ''
    method = ''
    conclusion = ''
    result = ''

    lines, pred = make_skimlit_predictions(abstract, model, tokenizer, label_encoder)
    # pred, lines = make_predictions(abstract)

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
    
    # creating model, tokenizer and labelEncoder
    cnt = 0
    if cnt == 0:
        skimlit_model, tokenizer, label_encoder = create_utils(MODEL_PATH, TOKENIZER_PATH, LABEL_ENOCDER_PATH, EMBEDDING_FILE_PATH)
        cnt = 1
    
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
            objective, background, methods, conclusion, result = model_prediction(abstract, skimlit_model, tokenizer, label_encoder)
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
