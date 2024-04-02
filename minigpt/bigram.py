# Inspired from Andrej Karpathy bigram model 

import jax
import jax.numpy as jnp
from jax import random, jit, grad
import haiku as hk
import optax
from functools import partial

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2

# Initialize PRNG
key = random.PRNGKey(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Assuming `stoi` is a dictionary mapping characters to integer indices
encoded_text = jnp.array([stoi[c] for c in text if c in stoi], dtype=jnp.int32)


# Split the encoded text into training and validation sets
split_index = int(0.9 * len(encoded_text))
train_data = encoded_text[:split_index]
val_data = encoded_text[split_index:]

# Define the bigram model
class BigramModel(hk.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, inputs):
        # Initialize the embedding table with shape [vocab_size, vocab_size]
        # Each row effectively acts as the logits for predicting the next token
        embedding_table = hk.get_parameter("embedding_table", shape=[self.vocab_size, self.vocab_size], 
                                           init=hk.initializers.RandomNormal(stddev=0.01))
        # Use advanced indexing to lookup the logits for the next token based on the current token
        logits = embedding_table[inputs]
        return logits

# Transform the model to be used with Haiku
def forward_pass(inputs):
    model = BigramModel(vocab_size=vocab_size)
    return model(inputs)

sequence_length = block_size

# Randomly generate some input data
inputs = jax.random.randint(jax.random.PRNGKey(42), (batch_size, sequence_length), 0, vocab_size)
targets = jax.random.randint(jax.random.PRNGKey(43), (batch_size, sequence_length), 0, vocab_size)

# Initialize the model
model = hk.transform(forward_pass)
params = model.init(jax.random.PRNGKey(42), inputs)

# Define the loss function
def loss_fn(params, inputs, targets):
    logits = model.apply(params, None, inputs)
    # Use softmax cross entropy as the loss
    loss = optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(targets, vocab_size))
    return jnp.mean(loss)

# Example of updating the model parameters
def update(params, inputs, targets, learning_rate=0.01):
    grads = jax.grad(loss_fn)(params, inputs, targets)
    updates, new_opt_state = optax.sgd(learning_rate).update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Initialize optimizer state
opt_state = optax.sgd(learning_rate=0.01).init(params)

print_every = 100  # Print the loss every 100 iterations
max_iters = 10000
for iter_num in range(max_iters):
    # Update the model parameters and optimizer state
    params, opt_state = update(params, inputs, targets, learning_rate=0.01)
    
    # Optionally, calculate the loss to monitor progress
    if iter_num % print_every == 0 or iter_num == max_iters - 1:
        current_loss = loss_fn(params, inputs, targets)
        print(f"Iteration {iter_num}, Loss: {current_loss:.4f}")

import numpy as np

def generate(model, params, start_idx, max_new_tokens, rng_key):
    """
    Generate text from the model.

    Parameters:
    - model: The Haiku transformed model.
    - params: The parameters of the model.
    - start_idx: The starting index (integer) or array of indices to begin generation.
    - max_new_tokens: The maximum number of new tokens to generate.
    - rng_key: The JAX random key.

    Returns:
    - A list of generated token indices.
    """
    generated_tokens = [start_idx] if np.isscalar(start_idx) else list(start_idx)
    current_input = np.array([start_idx]) if np.isscalar(start_idx) else np.array(start_idx[-1:])

    for _ in range(max_new_tokens):
        # Convert current input to JAX array
        current_input_jax = jnp.array(current_input).reshape(1, -1)
        # Predict the logits for the next token
        logits = model.apply(params, rng_key, current_input_jax)
        # Use the logits from the last token position to sample the next token
        logits_last_token = logits[:, -1, :]
        probs = jax.nn.softmax(logits_last_token)
        next_token = jax.random.categorical(rng_key, probs[0]).item()
        # Append the sampled token to the generated sequence
        generated_tokens.append(next_token)
        # Update the current input with the sampled token
        current_input = [next_token]

    return generated_tokens

# Example usage
rng_key = jax.random.PRNGKey(23)
# start_idx = [stoi['H']]  # Assuming 'H' is a character in your vocabulary and stoi is your char to index mapping
start_idx = [stoi['a']]
max_new_tokens = 100  # Generate 100 tokens

# Assuming 'model' and 'params' are already defined and initialized
generated_token_indices = generate(model, params, start_idx, max_new_tokens, rng_key)
generated_text = ''.join([itos[idx] for idx in generated_token_indices])  # Assuming itos is your index to char mapping

print(generated_text)
