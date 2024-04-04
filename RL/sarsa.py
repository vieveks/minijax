import jax
import jax.numpy as jnp
from jax import random

# Environment setup (simplified grid world)
num_states = 5
num_actions = 2  # Assume two actions: 0 (left) and 1 (right)
transition_matrix = jnp.array([
    [1, 0],  # State 0 transitions to state 1 if action is right, else loops back
    [2, 0],  # State 1 transitions to state 2 if action is right, else goes to state 0
    [3, 1],  # etc.
    [4, 2],
    [4, 3]
])

# Reward for each state-action pair (simplified)
rewards = jnp.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 1],  # Reward only when moving right from state 3 to 4
    [0, 0]
])

# Q-learning settings
gamma = 0.99  # Discount factor
alpha = 0.1   # Learning rate
epsilon = 0.1  # Exploration rate
max_steps = 100

# Initialize Q-table randomly
q_table = random.uniform(random.PRNGKey(0), (num_states, num_actions))

# SARSA update function
def sarsa_update(state, action, reward, next_state, next_action, q_table):
    next_max = jnp.max(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, next_action]
    td_error = td_target - q_table[state, action]
    q_table = q_table.at[state, action].set(q_table[state, action] + alpha * td_error)
    return q_table, next_action

# Training loop
num_episodes = 10
for episode in range(num_episodes):
    state = random.randint(random.PRNGKey(episode), (), 0, num_states)  # Start at a random state
    done = False
    count = 0
    while not done:
        count += 1
        # Epsilon-greedy action selection
        if random.uniform(random.PRNGKey(episode)) < epsilon:
            action = random.randint(random.PRNGKey(episode), (), 0, num_actions)
        else:
            action = jnp.argmax(q_table[state])

        # Take action
        next_state = transition_matrix[state, action]
        reward = rewards[state, action]

        # Select next action
        next_action = None
        if random.uniform(random.PRNGKey(episode)) < epsilon:
            next_action = random.randint(random.PRNGKey(episode), (), 0, num_actions)
        else:
            next_action = jnp.argmax(q_table[next_state])

        # Q-table update
        q_table, next_action = sarsa_update(state, action, reward, next_state, next_action, q_table)

        print(f"Episode {episode}, State {state}, Action {action}, Reward {reward}, Next state {next_state}, Next action {next_action}")

        # Transition to next state
        state = next_state
        if state == num_states - 1:  # Assuming the last state is terminal
            done = True

        if count >= max_steps:
            done = True

print("Trained Q-table:")
print(q_table)
