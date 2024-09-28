# data.py

import numpy as np
from config import config

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_data(text):
    # Create a character-level vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the text
    encoded_text = np.array([stoi[ch] for ch in text], dtype=np.int32)
    return encoded_text, stoi, itos, vocab_size

def get_batch(encoded_text, batch_size, seq_length):
    data_length = len(encoded_text)
    num_batches = data_length // (batch_size * seq_length)
    encoded_text = encoded_text[:num_batches * batch_size * seq_length]
    encoded_text = encoded_text.reshape((batch_size, -1))

    while True:
        for i in range(0, encoded_text.shape[1] - seq_length, seq_length):
            x = encoded_text[:, i:i+seq_length]
            y = encoded_text[:, i+1:i+seq_length+1]
            yield {'inputs': x, 'targets': y}
