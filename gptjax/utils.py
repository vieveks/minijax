# utils.py

import pickle

def save_model(params, path):
    with open(path, 'wb') as f:
        pickle.dump(params, f)

def load_model(path):
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return params
