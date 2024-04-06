import haiku as hk
import jax.numpy as jnp
import jax
from jax import random
import optax
from jax import grad , jit

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

# Data loading
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return jnp.array([stoi[c] for c in s], dtype=jnp.int32)

def decode(l):
    return ''.join([itos[i] for i in l])

data = encode(text)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(train_data[:100])
print(text[:100])
input('Press enter to continue...')

# now the batch 
def get_batch(split, train_data, val_data, block_size, batch_size, rng_key):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    data_size = len(data)
    
    # generate random indices
    ix = random.randint(rng_key, (batch_size,), 0, data_size - block_size)
    
    # create input and target arrays
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

# Example usage
block_size = 8             # your block size
batch_size = 4             # your batch size

rng_key = random.PRNGKey(42)   # random key for reproducibility

# get a batch of training data
rng_key, subkey = random.split(rng_key)
train_batch = get_batch('train', train_data, val_data, block_size, batch_size, subkey)

# get a batch of validation data
rng_key, subkey = random.split(rng_key)
val_batch = get_batch('val', train_data, val_data, block_size, batch_size, subkey)

# print the shapes of the batches
print(train_batch[1].shape)
print(train_batch[0].shape)

input('got the batch shapes, press enter to continue...')


class BigramLanguageModel(hk.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, idx, targets=None):
        # each token directly reads off the logits for the next token from a lookup table
        token_embedding_table = hk.Embed(self.vocab_size, self.vocab_size)
        logits = token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = jnp.reshape(logits, (B * T, C))
            targets = jnp.reshape(targets, (B * T,))
            loss =  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(targets, num_classes=self.vocab_size)))

        return logits, loss

    def generate(self, idx, max_new_tokens, rng_key):
    # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = jax.nn.softmax(logits, axis=-1)  # (B, C)
            # sample from the distribution
            rng_key, subkey = random.split(rng_key)
            idx_next = random.categorical(subkey, probs)  # (B,)
            idx_next = jnp.expand_dims(idx_next, axis=-1)  # (B, 1)
            # append sampled index to the running sequence
            idx = jnp.concatenate((idx, idx_next), axis=1)  # (B, T+1)
        return idx

def model_fn(vocab_size):
    def forward_fn(idx, targets=None):
        model = BigramLanguageModel(vocab_size)
        return model(idx, targets)
    return forward_fn

def generate_fn(tokens, rng_key, max_new_tokens):
    model = BigramLanguageModel(vocab_size)
    return model.generate(tokens, max_new_tokens, rng_key)


# Create the model
# Transform the functions
hk_model = hk.transform(lambda tokens, targets=None: model_fn(vocab_size)(tokens, targets))
hk_generate = hk.transform(lambda rng_key, tokens, max_new_tokens: generate_fn(tokens, rng_key, max_new_tokens))


# Initialize the model parameters
rng_key = jax.random.PRNGKey(42)
x, y = get_batch('train', train_data, val_data, block_size, batch_size, rng_key)
params = hk_model.init(rng_key, x, y)

# Apply the model
logits, loss = hk_model.apply(params, rng_key, x, y)

print(f'logits are {logits}')
print(f'loss is {loss}')

input('now generating the text proceed? ')
# generate from the model sample before trainig 
# context = jnp.array([[0]], dtype=jnp.int32)
# generated_indices = hk_generate.apply(params, None, context, subkey ,50)
# generated_text = decode(generated_indices[0].tolist())
# print(generated_text)

# Training loop
input('Starting the training proceed ?')

@jit
def train_step(params, rng_key, x, y, optimizer_state):
    """Perform a single training step."""
    def loss_fn(params, x, y):
        """Loss function to be minimized."""
        logits, loss = hk_model.apply(params, rng_key, x, y)
        return loss

    grads = grad(loss_fn)(params, x, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, optimizer_state

# Evaluation
# @jit
def evaluate(params, rng_key, data, block_size, batch_size):
    """Evaluate the model's performance on validation data."""
    losses = []
    for _ in range(eval_iters):
        rng_key, subkey = random.split(rng_key)
        x, y = get_batch('val', data, data, block_size, batch_size, subkey)
        _, loss = hk_model.apply(params, rng_key, x, y)
        losses.append(loss)
    return jnp.mean(jnp.array(losses))

# evaluate = jax.jit(evaluate, static_argnums=(2, 3))
# Initialize the model parameters
rng_key = jax.random.PRNGKey(42)
x, y = get_batch('train', train_data, val_data, block_size, batch_size, rng_key)
params = hk_model.init(rng_key, x, y)

# Create the optimizer
optimizer = optax.adam(learning_rate)
optimizer_state = optimizer.init(params)

#sample values u may change 
batch_size = 32
block_size = 8
max_iters = 1200
eval_interval = 40
learning_rate = 1e-2
eval_iters = 20

# Training loop
for iter in range(max_iters):
    rng_key, subkey = random.split(rng_key)
    x, y = get_batch('train', train_data, val_data, block_size, batch_size, subkey)
    params, optimizer_state = train_step(params, rng_key, x, y, optimizer_state)

    if iter % eval_interval == 0:
        val_loss = evaluate(params, rng_key, val_data, block_size, batch_size)
        print(f"Iteration {iter}: Validation Loss = {val_loss:.4f}")

# Generate text from the trained model
context = jnp.array([[0]], dtype=jnp.int32)
rng_key, subkey = random.split(rng_key)
generated_indices = hk_generate.apply(params, None, subkey, context,  max_new_tokens=50)
generated_text = decode(generated_indices[0].tolist())
print(generated_text)

