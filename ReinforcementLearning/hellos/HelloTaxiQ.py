import gym
import numpy as np

env = gym.make("Taxi-v2")

Q = np.zeros([env.observation_space.n, env.action_space.n])

for episode in range(1, 1001):
    alpha = 0.618
    gamma = 1.0
    G = 0
    reward = 0
    state = env.reset()
    done = False

    while done != True:
        action = np.argmax(Q[state])
        nState, reward, done, info = env.step(action)
        Q[state, action] += alpha * \
            (reward + gamma * np.max(Q[nState]) - Q[state, action])
        G += reward
        state = nState

    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))

env.render()
