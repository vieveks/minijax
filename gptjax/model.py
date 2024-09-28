# model.py

import haiku as hk
import jax
import jax.numpy as jnp
from config import config

class MultiHeadSelfAttention(hk.Module):
    def __init__(self, num_heads, head_size, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.head_size = head_size
        self.output_size = num_heads * head_size

    def __call__(self, x):
        batch_size, seq_length, embed_size = x.shape
        qkv = hk.Linear(3 * self.output_size)(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_size)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_length, head_size)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        scores = jnp.einsum('bhqd,bhkd->bhqk', queries, keys) / jnp.sqrt(self.head_size)
        mask = jnp.tril(jnp.ones((seq_length, seq_length)))
        scores = scores * mask - 1e10 * (1 - mask)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attention_output = jnp.einsum('bhqk,bhkd->bhqd', attention_weights, values)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.output_size)
        output = hk.Linear(embed_size)(attention_output)
        return output

class FeedForwardNetwork(hk.Module):
    def __init__(self, hidden_size, output_size, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(self, x):
        x = hk.Linear(self.hidden_size)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(self.output_size)(x)
        return x

class TransformerBlock(hk.Module):
    def __init__(self, num_heads, head_size, hidden_size, name=None):
        super().__init__(name=name)
        self.attention = MultiHeadSelfAttention(num_heads, head_size)
        self.ffn = FeedForwardNetwork(hidden_size, config['embed_size'])

    def __call__(self, x):
        # Multi-Head Self-Attention
        attn_output = self.attention(x)
        x = x + attn_output  # Residual connection
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = x + ffn_output  # Residual connection
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        return x

class GPTModel(hk.Module):
    def __init__(self, vocab_size, seq_length, num_layers, num_heads, head_size, name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_size = config['embed_size']
        self.token_embedding = hk.Embed(vocab_size=vocab_size, embed_dim=self.embed_size)
        self.position_embedding = hk.Embed(vocab_size=seq_length, embed_dim=self.embed_size)
        self.layers = [TransformerBlock(num_heads, head_size, config['hidden_size']) for _ in range(num_layers)]
        self.output_layer = hk.Linear(vocab_size)

    def __call__(self, x):
        batch_size = x.shape[0]
        positions = jnp.tile(jnp.arange(self.seq_length), (batch_size, 1))
        x = self.token_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        logits = self.output_layer(x)
        return logits
