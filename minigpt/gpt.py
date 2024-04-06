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
# block_size = 8
# batch_size = 4

rng_key = random.PRNGKey(42)
rng_key, subkey = random.split(rng_key)
train_batch = get_batch('train', train_data, val_data, block_size, batch_size, subkey)
rng_key, subkey = random.split(rng_key)
val_batch = get_batch('val', train_data, val_data, block_size, batch_size, subkey)
print(train_batch[1].shape)
print(train_batch[0].shape)
input('got the batch shapes, press enter to continue...')
rng_key = random.PRNGKey(42)   # random key for reproducibility

# get a batch of training data
rng_key, subkey = random.split(rng_key)
train_batch = get_batch('train', train_data, val_data, block_size, batch_size, subkey)

# get a batch of validation data
rng_key, subkey = random.split(rng_key)
val_batch = get_batch('val', train_data, val_data, block_size, batch_size, subkey)

input('got the batch shapes, press enter to continue...')

# Hyperparameters
batch_size = 32
block_size = 8
# max_iters = 3000
# eval_interval = 300
max_iters = 50
eval_interval = 5
learning_rate = 1e-2
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2

class GPTLanguageModel(hk.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

    def __call__(self, idx, targets=None):
        token_embedding_table = hk.Embed(self.vocab_size, self.n_embd)
        position_embedding_table = hk.Embed(self.block_size, self.n_embd)
        x = token_embedding_table(idx) + position_embedding_table(jnp.arange(idx.shape[1]))
        x = hk.dropout(hk.next_rng_key(), self.dropout, x)

        for _ in range(self.n_layer):
            x = hk.MultiHeadAttention(self.n_head, self.n_embd // self.n_head, w_init_scale=1.0)(x, x, x)
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
            x = hk.Linear(4 * self.n_embd)(x)
            x = jax.nn.gelu(x)
            x = hk.Linear(self.n_embd)(x)
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        logits = hk.Linear(self.vocab_size)(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = jnp.reshape(logits, (B * T, C))
            targets = jnp.reshape(targets, (B * T,))
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(targets, num_classes=self.vocab_size)))

        return logits, loss

    # def generate(self, idx, max_new_tokens, rng_key):
    #     for _ in range(max_new_tokens):
    #         logits, _ = self(idx)
    #         logits = logits[:, -1, :]
    #         probs = jax.nn.softmax(logits, axis=-1)
    #         rng_key, subkey = random.split(rng_key)
    #         idx_next = random.categorical(subkey, probs)
    #         idx_next = jnp.expand_dims(idx_next, axis=-1)
    #         idx = jnp.concatenate((idx, idx_next), axis=1)
    #     return idx

    def generate(self, idx, max_new_tokens, rng_key):
        for _ in range(max_new_tokens):
            rng_key, subkey = random.split(rng_key)  # Splitting the rng_key for each iteration
            logits, _ = self(idx, rng_key=subkey)  # Pass the subkey to the model call
            logits = logits[:, -1, :]
            probs = jax.nn.softmax(logits, axis=-1)
            idx_next = random.categorical(subkey, probs)
            idx_next = jnp.expand_dims(idx_next, axis=-1)
            idx = jnp.concatenate((idx, idx_next), axis=1)
        return idx


def model_fn(vocab_size, n_embd, n_head, n_layer, block_size, dropout):
    def forward_fn(idx, targets=None):
        model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
        return model(idx, targets)
    return forward_fn

def generate_fn( rng_key,tokens, max_new_tokens, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    return model.generate(tokens, max_new_tokens, rng_key)

# Create the model
hk_model = hk.transform(lambda tokens, targets=None: model_fn(vocab_size, n_embd, n_head, n_layer, block_size, dropout)(tokens, targets))
hk_generate = hk.transform(lambda rng_key, tokens, max_new_tokens: generate_fn(tokens, rng_key, max_new_tokens, vocab_size, n_embd, n_head, n_layer, block_size, dropout))

# Initialize the model parameters
rng_key = jax.random.PRNGKey(42)
x, y = get_batch('train', train_data, val_data, block_size, batch_size, rng_key)
params = hk_model.init(rng_key, x, y)

# Apply the model
logits, loss = hk_model.apply(params, rng_key, x, y)
print(f'logits are {logits}')
print(f'loss is {loss}')

input('now generating the text proceed? ')

# Training loop
input('Starting the training proceed ?')

@jit
def train_step(params, rng_key, x, y, optimizer_state):
    def loss_fn(params, x, y):
        logits, loss = hk_model.apply(params, rng_key, x, y)
        return loss

    grads = grad(loss_fn)(params, x, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, optimizer_state

def evaluate(params, rng_key, data, block_size, batch_size):
    losses = []
    for _ in range(eval_iters):
        rng_key, subkey = random.split(rng_key)
        x, y = get_batch('val', data, data, block_size, batch_size, subkey)
        _, loss = hk_model.apply(params, rng_key, x, y)
        losses.append(loss)
    return jnp.mean(jnp.array(losses))

# Initialize the model parameters
rng_key = jax.random.PRNGKey(42)
x, y = get_batch('train', train_data, val_data, block_size, batch_size, rng_key)
params = hk_model.init(rng_key, x, y)

# Create the optimizer
optimizer = optax.adam(learning_rate)
optimizer_state = optimizer.init(params)

# Training loop
for iter in range(max_iters):
    rng_key, subkey = random.split(rng_key)
    x, y = get_batch('train', train_data, val_data, block_size, batch_size, subkey)
    params, optimizer_state = train_step(params, rng_key, x, y, optimizer_state)

    if iter % eval_interval == 0:
        val_loss = evaluate(params, rng_key, val_data, block_size, batch_size)
        print(f"Iteration {iter}: Validation Loss = {val_loss:.4f}")

input(f'training done, Proceed ? ')        

# Generate text from the trained model
context = jnp.array([[0]], dtype=jnp.int32)
rng_key, subkey = random.split(rng_key)
generated_indices = hk_generate.apply(params, rng_key, None, context, max_new_tokens=500)
generated_text = decode(generated_indices[0].tolist())
print(generated_text)