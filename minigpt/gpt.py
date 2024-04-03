import jax
import jax.numpy as jnp
from jax import random
import haiku as hk
import optax
from functools import partial
import numpy as np

# Assuming necessary imports and data preprocessing as before
batch_size = 32
block_size = 8
# max_iters = 3000
max_iters = 50
eval_interval = 50
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

# Define a simplified version of the Multi-Head Attention used in GPT-2
class MultiHeadAttention(hk.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads

        self.wq = hk.Linear(self.d_model)
        self.wk = hk.Linear(self.d_model)
        self.wv = hk.Linear(self.d_model)

        self.dense = hk.Linear(self.d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        depth = q.shape[-1]
        logits = matmul_qk / jnp.sqrt(depth)
        attention_weights = jax.nn.softmax(logits, axis=-1)
        output = jnp.matmul(attention_weights, v)
        return output

    def __call__(self, v, k, q):
        batch_size = q.shape[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Assuming you have a function for scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(q, k, v)  # Shape: (batch_size, num_heads, seq_length, depth)

        # Step 1: Concatenate the heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # Step 2: Optionally apply a dense layer if you want to project back to original dimensionality
        # This step might be included depending on your specific implementation and needs
        attn_output = self.dense(attn_output)  # Shape: (batch_size, seq_length, d_model)

        return attn_output

# Define the Transformer block
class TransformerBlock(hk.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = hk.Sequential([
            hk.Linear(d_model * 4),
            jax.nn.relu,
            hk.Linear(d_model),
        ])
        self.layernorm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.layernorm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x):
        attn_output = self.mha(x, x, x)  # Simplified for illustration
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Define the GPT-2 Model
class GPT2Model(hk.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = hk.Embed(vocab_size, d_model)
        self.pos_encoding = hk.get_parameter("pos_encoding", [1, block_size, d_model], init=hk.initializers.RandomNormal())
        self.transformer_blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.final_layer = hk.Linear(vocab_size)

    def __call__(self, x):
        seq_length = x.shape[1]
        positions = jnp.arange(seq_length)[None, :]
        
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_length, :]
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        logits = self.final_layer(x)
        return logits

# Transform the model to be used with Haiku
def gpt2_forward_pass(inputs):
    model = GPT2Model(vocab_size=vocab_size, d_model=512, num_heads=8, num_layers=12)
    return model(inputs)

sequence_length = 8

# Randomly generate some input data
inputs = jax.random.randint(jax.random.PRNGKey(42), (batch_size, sequence_length), 0, vocab_size)
targets = jax.random.randint(jax.random.PRNGKey(43), (batch_size, sequence_length), 0, vocab_size)

print("Input:\n", inputs[0])
print("Targets:\n", targets[0])
input('Press Enter to continue...')

# Initialize the model
gpt2_model = hk.transform(gpt2_forward_pass)
params = gpt2_model.init(jax.random.PRNGKey(42), inputs)

# Define the loss function
def gpt2_loss_fn(params, inputs, targets):
    logits = gpt2_model.apply(params, None, inputs)
    # Use softmax cross entropy as the loss
    loss = optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(targets, vocab_size))
    return jnp.mean(loss)

# Example of updating the model parameters
def gpt2_update(params, inputs, targets, learning_rate=0.01):
    grads = jax.grad(gpt2_loss_fn)(params, inputs, targets)
    updates, new_opt_state = optax.sgd(learning_rate).update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Initialize optimizer state for GPT-2
opt_state = optax.sgd(learning_rate=0.01).init(params)
print_every = 10

# Example training loop for GPT-2
for iter_num in range(max_iters):
    params, opt_state = gpt2_update(params, inputs, targets, learning_rate=0.01)
    
    if iter_num % print_every == 0 or iter_num == max_iters - 1:
        current_loss = gpt2_loss_fn(params, inputs, targets)
        print(f"Iteration {iter_num}, Loss: {current_loss:.4f}")  

# generate the text

def generate(model, params, start_sequence, max_new_tokens, temperature=1.0):
    generated_tokens = encode(start_sequence)
    
    for _ in range(max_new_tokens):
        # Convert the generated tokens to a JAX array
        input_tokens = jnp.array(generated_tokens[-block_size:], dtype=jnp.int32)[None, :]
        
        # Get the model predictions (logits)
        logits = model.apply(params, None, input_tokens)
        
        # Apply temperature scaling to the logits
        logits = logits[:, -1, :] / temperature
        
        # Convert logits to probabilities using softmax
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Sample the next token from the probability distribution
        next_token = jax.random.categorical(jax.random.PRNGKey(0), probs).item()
        
        # Append the sampled token to the generated sequence
        generated_tokens.append(next_token)
    
    # Decode the generated tokens back to text
    generated_text = decode(generated_tokens)
    
    return generated_text

start_sequence = "The quick brown fox"
max_new_tokens = 20
temperature = 1.0

generated_text = generate(gpt2_model, params, start_sequence, max_new_tokens, temperature)
print(generated_text)