import gym
import numpy as np

env = gym.make("MsPacman-v0")

print("Action space", env.action_space.n)
print("Actions", env.env.get_action_meanings())
print("Observation space", env.observation_space)

counter = 0
state = env.reset()
reward = None
done = False

while done is False:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    counter += 1

    print(counter, action, reward, done)
    env.render()
