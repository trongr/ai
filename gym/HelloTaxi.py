import gym
import numpy as np

env = gym.make("Taxi-v2")

print("observation_space", env.observation_space.n)
print("action_space", env.action_space.n)
print("action_space", env.action_space)

state = env.reset()
counter = 0
reward = None
done = False

# while done is False:
while reward != 20:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    counter += 1

    print(counter, action, reward, done)
    env.render()
