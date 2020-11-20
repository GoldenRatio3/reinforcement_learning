import gym
import os
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

