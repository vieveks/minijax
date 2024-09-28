# train.py

import jax
import jax.numpy as jnp
import optax
import haiku as hk
from model import GPTModel
from data import load_data, preprocess_data, get_batch
from utils import save_model
from config import config

def cross_entropy_loss(logits, targets):
    one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_targets)
    return loss.mean()

def train_model():
    # Load and preprocess data
    text = load_data('data/train.txt')
    encoded_text, stoi, itos, vocab_size = preprocess_data(text)
    batch_generator = get_batch(encoded_text, config['batch_size'], config['seq_length'])

    # Initialize model and optimizer
    def forward_fn(x):
        model = GPTModel(vocab_size, config['seq_length'], config['num_layers'],
                         config['num_heads'], config['head_size'])
        return model(x)

    model = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    sample_batch = next(batch_generator)
    params = model.init(rng, sample_batch['inputs'])

    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, batch):
        def loss_fn(params):
            logits = model.apply(params, rng, batch['inputs'])
            loss = cross_entropy_loss(logits, batch['targets'])
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    for epoch in range(config['num_epochs']):
        batch = next(batch_generator)
        params, opt_state, loss = update(params, opt_state, batch)
        if epoch % config['log_every'] == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
            save_model(params, f'checkpoints/model_epoch_{epoch}.pkl')
