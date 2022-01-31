import numpy as np
from spacy.lang.en import English
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import torch
import torch.nn.functional as F

from Dataset import SkimlitDataset

# nltk.download("stopwords")
# STOPWORDS = stopwords.words("english")
# porter = PorterStemmer()

def download_stopwords():
    nltk.download("stopwords")
    STOPWORDS = stopwords.words("english")
    porter = PorterStemmer()
    return STOPWORDS, porter

def preprocess(text, stopwords):
    """Conditional preprocessing on our text unique to our task."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub("", text)

    # Remove words in paranthesis
    text = re.sub(r"\([^)]*\)", "", text)

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    return text

def spacy_function(abstract):
    
  # setup English sentence parser
  nlp = English()

  # create sentence splitting pipeline object
  sentencizer = nlp.create_pipe("sentencizer")

  # add sentence splitting pipeline object to sentence parser
  nlp.add_pipe('sentencizer')
    
  # create "doc" of parsed sequences, change index for a different abstract
  doc = nlp(abstract) 

  # return detected sentences from doc in string type (not spaCy token type)
  abstract_lines = [str(sent) for sent in list(doc.sents)]
    
  return abstract_lines
    
# ---------------------------------------------------------------------------------------------------------------------------

def model_prediction(model, dataloader):
  """Prediction step."""
  # Set model to eval mode
  model.eval()
  y_trues, y_probs = [], []
  # Iterate over val batches
  for i, batch in enumerate(dataloader):
    # Forward pass w/ inputs
    # batch = [item.to(.device) for item in batch]  # Set device
    inputs = batch
    z = model(inputs)
    # Store outputs
    y_prob = F.softmax(z, dim=1).detach().cpu().numpy()
    y_probs.extend(y_prob)
  return np.vstack(y_probs)

# ---------------------------------------------------------------------------------------------------------------------------

def make_skimlit_predictions(text, model, tokenizer, label_encoder): # embedding path
  # getting all lines seprated from abstract
  abstract_lines = list()
  abstract_lines = spacy_function(text)  
    
  # Get total number of lines
  total_lines_in_sample = len(abstract_lines)

  # Go through each line in abstract and create a list of dictionaries containing features for each line
  sample_lines = []
  for i, line in enumerate(abstract_lines):
    sample_dict = {}
    sample_dict["text"] = str(line)
    sample_dict["line_number"] = i
    sample_dict["total_lines"] = total_lines_in_sample - 1
    sample_lines.append(sample_dict)

  # converting sample line list into pandas Dataframe
  df = pd.DataFrame(sample_lines)
  
  # getting stopword
  STOPWORDS, porter = download_stopwords()

  # applying preprocessing function to lines
  df.text = df.text.apply(lambda x: preprocess(x, STOPWORDS))

  # converting texts into numberical sequences
  text_seq = tokenizer.texts_to_sequences(texts=df['text'])

  # creating Dataset
  dataset = SkimlitDataset(text_seq=text_seq, line_num=df['line_number'], total_line=df['total_lines'])

  # creating dataloader
  dataloader = dataset.create_dataloader(batch_size=2)

  # Preparing embedings
#   embedding_matrix = get_embeddings(embeding_path, tokenizer, 300)

  # creating model
#   model = SkimlitModel(embedding_dim=300, vocab_size=len(tokenizer), hidden_dim=128, n_layers=3, linear_output=128, num_classes=len(label_encoder), pretrained_embeddings=embedding_matrix)

  # loading model weight
#   model.load_state_dict(torch.load('/content/drive/MyDrive/Datasets/SkimLit/skimlit-pytorch-1/skimlit-model-final-1.pt', map_location='cpu'))

  # setting model into evaluation mode
  model.eval()

  # getting predictions 
  y_pred = model_prediction(model, dataloader)

  # converting predictions into label class
  pred = y_pred.argmax(axis=1)
  pred = label_encoder.decode(pred)

  return abstract_lines, pred