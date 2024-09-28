# evaluate.py

import jax
import jax.numpy as jnp
import haiku as hk
from model import GPTModel
from data import preprocess_data
from utils import load_model
from config import config

def generate_text(params, seed_text, stoi, itos, vocab_size, length=100):
    def forward_fn(x):
        model = GPTModel(vocab_size, config['seq_length'], config['num_layers'],
                         config['num_heads'], config['head_size'])
        return model(x)

    model = hk.transform(forward_fn)

    # Initialize input sequence
    input_seq = [stoi[ch] for ch in seed_text]
    generated = input_seq.copy()

    for _ in range(length):
        x = jnp.array([generated[-config['seq_length']:]], dtype=jnp.int32)
        logits = model.apply(params, None, x)
        next_token_logits = logits[0, -1]
        next_token = jnp.argmax(next_token_logits)
        generated.append(int(next_token))

    generated_text = ''.join([itos[idx] for idx in generated])
    return generated_text

def evaluate_model():
    # Load model parameters
    params = load_model('checkpoints/model_epoch_latest.pkl')

    # Load and preprocess data to get vocab
    text = load_data('data/train.txt')
    _, stoi, itos, vocab_size = preprocess_data(text)

    seed_text = "Once upon a time"
    generated_text = generate_text(params, seed_text, stoi, itos, vocab_size)
    print(generated_text)
