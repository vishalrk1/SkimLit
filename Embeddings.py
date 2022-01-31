import torch
import numpy as np

def load_glove_embeddings(embeddings_file):
    """Load embeddings from a file."""
    embeddings = {}
    with open(embeddings_file, "r", encoding="utf8") as fp:
        for index, line in enumerate(fp):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings

def make_embeddings_matrix(embeddings, word_index, embedding_dim):
    """Create embeddings matrix to use in Embedding layer."""
    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_embeddings(embedding_file_path, tokenizer, embedding_dim):
    glove_embeddings = load_glove_embeddings(embeddings_file=embedding_file_path)
    embedding_matrix = make_embeddings_matrix(embeddings=glove_embeddings, word_index=tokenizer.token_to_index, embedding_dim=embedding_dim)
    return embedding_matrix