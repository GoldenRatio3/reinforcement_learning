import gym
import os
import numpy as np
import random
from time import sleep

def print_frames(frames):
  for i, frame in enumerate(frames):
    os.system('clear')
    print(frame['frame'])
    print(f"Timestep: {i + 1}")
    print(f"State: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")
    sleep(.1)

env = gym.make("Taxi-v3").env

env.render() # debugging, prints the environment

print("Action space {}".format(env.action_space))
print("State space {}".format(env.observation_space))

state = env.encode(3, 1, 2, 0)
print("State: ", state)
env.s = state
env.render()
print(env.P[328]) # reward structure {action: [{probability, nextstate, reward, done}]}

# Solving without RL (Brute force approach)
print()
print('Solving wo/ RL (Brute force approach)')
print('-'*20)
env.s = 328
epochs = 0
penalties, reward = 0, 0

frames = []

done = False

while not done:
  action = env.action_space.sample() # picks random action
  state, reward, done, info = env.step(action)

  if reward == -10:
    penalties += 1

  frames.append({
    'frame': env.render(mode='ansi'),
    'state': state,
    'action': action,
    'reward': reward
  })

  epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

value = input("Printing frames (will clear screen) (Y/n)")

if value == 'Y':
  print_frames(frames)

# Solving with RL (Q-Learning)
print()
print('Solving w/ Rl (Q-Learning)')
print('-'*20)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# for plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
  state = env.reset()
  epochs, penalties, reward = 0, 0, 0
  done = False

  while not done:
    if random.uniform(0, 1) < epsilon:
      action = env.action_space.sample() # explore action space
    else:
      action = np.argmax(q_table[state]) # exploit learned values

    next_state, reward, done, info = env.step(action)

    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state])

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state, action] = new_value

    if reward == -10:
      penalties += 1

    state = next_state
    epochs += 1

  if i % 100 == 0:
    os.system('clear')
    print(f"Episode: {i}")

print("Training finished.\n")

# Evaluate agents performance after Q-Learning
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
  state = env.reset()
  epochs, penalties, reward = 0, 0, 0
  done = False

  while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
      penalties += 1

    epochs += 1

  total_penalties += penalties
  total_epochs += epochs

print(f"Results after {episodes} episodes")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")





