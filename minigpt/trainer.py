import time
from collections import defaultdict

import jax
import jax.numpy as jnp
from jax import lax, random
import haiku as hk
import optax

from mingpt.utils import CfgNode as CN

class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
        else:
            self.device = config.device

        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = iter(self.train_dataset)

        rng = jax.random.PRNGKey(42)
        model_params = self.model.init(rng, jnp.ones([config.batch_size, 1], dtype=jnp.int32))
        opt_state = self.optimizer.init(model_params)

        @jax.jit
        def train_step(params, opt_state, batch):
            def loss_fn(params, batch):
                logits, loss = self.model.apply(params, batch, is_training=True)
                return loss

            grads = jax.grad(loss_fn)(params, batch)
            clipped_grads = optax.clip_grads(grads, config.grad_norm_clip)
            updates, opt_state = self.optimizer.update(clipped_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        for _ in range(config.max_iters):
            try:
                batch = next(train_loader)
            except StopIteration:
                train_loader = iter(self.train_dataset)
                batch = next(train_loader)

            batch = jnp.array(batch, dtype=jnp.int32)
            model_params, opt_state = train_step(model_params, opt_state, batch)

            self.trigger_callbacks('on_batch_end')

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break