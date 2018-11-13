import numpy as np
import gym
import random

env = gym.make("Taxi-v2")
env.render()

# The full state of the game at any point (the map, the drop off location, the
# position of the taxi and the passenger) is encoded into one of the 500 states.
ACTION_SIZE = env.action_space.n  # 6
STATE_SIZE = env.observation_space.n  # 500

qtable = np.zeros((STATE_SIZE, ACTION_SIZE))

TOTAL_EPISODES = 50000        # Total episodes
TOTAL_TEST_EPISODES = 100     # Total test episodes
MAX_STEPS = 99                # Max steps per episode

LEARNING_RATE = 0.7           # Learning rate
GAMMA = 0.618                 # Discounting rate

epsilon = 1.0                 # Exploration rate
MAX_EPSILON = 1.0             # Exploration probability at start
MIN_EPSILON = 0.01            # Minimum exploration probability
DECAY_RATE = 0.01             # Exponential decay rate for exploration prob

RewardHistory = []

for episode in range(TOTAL_EPISODES):
    state = env.reset()
    step = 0
    done = False
    EpisodeReward = 0

    for step in range(MAX_STEPS):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        nState, reward, done, info = env.step(action)
        qtable[state, action] = (1 - LEARNING_RATE) * qtable[state, action] + \
            LEARNING_RATE * (reward + GAMMA * np.max(qtable[nState, :]))

        EpisodeReward += reward
        state = nState

        if done == True:
            print("Ep: {}. Step: {}. Ep reward: {}. Avg reward: {}".format(
                episode,
                step,
                EpisodeReward,
                sum(RewardHistory) / (1 + len(RewardHistory))))
            env.render()
            break

    # Reduce epsilon because we need less and less exploration
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
        np.exp(-DECAY_RATE * episode)
    RewardHistory.append(EpisodeReward)

print(qtable)

for episode in range(TOTAL_TEST_EPISODES):
    state = env.reset()
    step = 0
    done = False

    print("============================================================")
    print("Ep: {}".format(episode))

    for step in range(MAX_STEPS):
        action = np.argmax(qtable[state, :])
        nState, reward, done, info = env.step(action)
        env.render()

        if done:
            print("Total steps taken: {}".format(step))
            print("============================================================")
            break

        state = nState

env.close()
