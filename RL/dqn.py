import jax
import jax.numpy as jnp
import haiku as hk
import optax
import gymnasium as gym
import numpy as np

# Define the Q-network
class QNetwork(hk.Module):
    def __init__(self, num_actions, name=None):
        super().__init__(name=name)
        self.num_actions = num_actions

    def __call__(self, observations):
        x = hk.Flatten()(observations)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        q_values = hk.Linear(self.num_actions)(x)
        return q_values

# Define the DQN agent
class DQNAgent:
    def __init__(self, env, gamma=0.99, learning_rate=1e-3, batch_size=32, buffer_size=10000):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.q_network = hk.transform(lambda obs: QNetwork(env.action_space.n)(obs))
        self.target_q_network = hk.transform(lambda obs: QNetwork(env.action_space.n)(obs))

        self.q_network_params = self.q_network.init(jax.random.PRNGKey(42), env.reset())
        self.target_q_network_params = self.q_network_params

        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.q_network_params)

        self.replay_buffer = []

    def select_action(self, observation, epsilon=0.1):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_network.apply(self.q_network_params, observation)
            return jnp.argmax(q_values).item()

    def update_target_network(self):
        self.target_q_network_params = self.q_network_params

    def train_step(self, batch):
        observations, actions, rewards, next_observations, dones = batch

        def loss_fn(params):
            q_values = self.q_network.apply(params, observations)
            q_values_selected = q_values[jnp.arange(len(actions)), actions]

            target_q_values = self.target_q_network.apply(self.target_q_network_params, next_observations)
            max_target_q_values = jnp.max(target_q_values, axis=1)
            target_q_values_expected = rewards + self.gamma * (1 - dones) * max_target_q_values

            loss = jnp.mean((q_values_selected - target_q_values_expected) ** 2)
            return loss

        grads = jax.grad(loss_fn)(self.q_network_params)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.q_network_params, updates)

        self.q_network_params = new_params
        self.opt_state = new_opt_state

        return loss_fn(new_params)

    def train(self, num_episodes, epsilon_decay=0.999, min_epsilon=0.1):
        epsilon = 1.0
        for episode in range(num_episodes):
            observation = self.env.reset()
            done = False
            while not done:
                action = self.select_action(observation, epsilon)
                next_observation, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((observation, action, reward, next_observation, done))

                if len(self.replay_buffer) > self.buffer_size:
                    self.replay_buffer.pop(0)

                observation = next_observation

                if len(self.replay_buffer) >= self.batch_size:
                    batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)
                    batch = list(zip(*batch))
                    loss = self.train_step(batch)
                    print(f"Episode {episode}, Loss: {loss:.4f}")

            epsilon = max(epsilon * epsilon_decay, min_epsilon)
            self.update_target_network()

# Example usage
env = gym.make("CartPole-v1")
agent = DQNAgent(env)
agent.train(num_episodes=1000)
