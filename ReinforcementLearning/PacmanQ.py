import gym
import numpy as np

env = gym.make("MsPacman-v0")

print("Action space", env.action_space.n)
print("Actions", env.env.get_action_meanings())
print("Observation space", env.observation_space)

Q = np.zeros([210 * 160 * 3, env.action_space.n])

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
        # This isn't right: can't index into Q with state cause state is 3
        # dimensional.
        Q[state, action] += alpha * \
            (reward + gamma * np.max(Q[nState]) - Q[state, action])
        G += reward
        state = nState
        env.render()
        print("Action: {}. Reward: {}. Return: {}".format(action, reward, G))

    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))
