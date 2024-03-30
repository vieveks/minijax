"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) https://github.com/karpathy/minGPT/mingpt/model.py Andrej Karpathys minigpt code implementation in pytorch 

"""

import jax
import jax.numpy as jnp
from jax import lax, random
import haiku as hk

def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))

class CausalSelfAttention(hk.Module):
    def __init__(self, n_head, n_embd, attn_pdrop, resid_pdrop, name=None):
        super().__init__(name=name)
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

    def __call__(self, x, mask, is_training):
        B, T, C = x.shape
        q, k, v = jnp.split(hk.Linear(3 * self.n_embd)(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 3, 1)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        attn_scores = jnp.matmul(q, k, transpose_b=True) / jnp.sqrt(k.shape[-1])
        attn_scores = attn_scores + mask

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_probs = hk.dropout(hk.next_rng_key(), attn_probs, rate=self.attn_pdrop, is_training=is_training)

        attn_output = jnp.matmul(attn_probs, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)
        attn_output = hk.dropout(hk.next_rng_key(), attn_output, rate=self.resid_pdrop, is_training=is_training)

        return hk.Linear(self.n_embd)(attn_output)

class Block(hk.Module):
    def __init__(self, n_head, n_embd, attn_pdrop, resid_pdrop, name=None):
        super().__init__(name=name)
        self.ln_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.attn = CausalSelfAttention(n_head, n_embd, attn_pdrop, resid_pdrop)
        self.ln_2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x, mask, is_training):
        x = x + self.attn(self.ln_1(x), mask, is_training)
        x = x + hk.Sequential([
            hk.Linear(4 * x.shape[-1]), gelu,
            hk.dropout(hk.next_rng_key(), rate=self.resid_pdrop, is_training=is_training),
            hk.Linear(x.shape[-1])
        ])(self.ln_2(x))
        return x

class GPT(hk.Module):
    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size, embd_pdrop, attn_pdrop, resid_pdrop, name=None):
        super().__init__(name=name)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embd_pdrop = embd_pdrop

        self.wte = hk.Embed(vocab_size, n_embd)
        self.wpe = hk.Embed(block_size, n_embd)
        self.drop = hk.dropout(hk.next_rng_key(), rate=embd_pdrop, is_training=True)
        self.blocks = [Block(n_head, n_embd, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        self.ln_f = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.lm_head = hk.Linear(vocab_size)

    def __call__(self, idx, targets=None, is_training=True):
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        pos = jnp.arange(0, T, dtype=jnp.long)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, is_training)

        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        for block in self.blocks:
            x = block(x, mask, is_training)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = jnp.mean(jax.nn.cross_entropy_loss_simple(logits.reshape(-1, self.vocab_size), targets.reshape(-1)))

        return logits, loss

    @jax.jit
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        B, T = idx.shape
        for _ in range(max_new_tokens):
            idx_cond = idx if T <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond, is_training=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_k_values, top_k_indices = jax.lax.top_k(logits, k=top_k)
                logits = jax.ops.index_update(logits, jnp.ones_like(logits.shape, dtype=bool), -jnp.inf)
                for i in range(logits.shape[0]):
                    logits = jax.ops.index_update(logits, top_k_indices[i], top_k_values[i])
            probs = jax.nn.softmax(logits, axis=-1)
            idx_next = random.categorical(hk.next_rng_key(), probs)
            idx = jnp.concatenate([idx, idx_next[:, None]], axis=1)
            T += 1
        return idx