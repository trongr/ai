import numpy as np
import gym
import random
from bokeh.plotting import figure, output_file, show

env = gym.make("FrozenLake-v0")

stateSize = env.observation_space.n
actionSize = env.action_space.n

qtable = np.zeros((stateSize, actionSize))

TOTAL_EPISODES = 15000  # Total episodes
LEARNING_RATE = 0.8  # Learning rate
MAX_STEPS = 99  # Max steps per episode
GAMMA = 0.95  # Discounting rate

epsilon = 1.0  # Exploration rate
MAX_EPSILON = 1.0  # Maximum exploration probability
MIN_EPSILON = 0.01  # Minimum exploration probability
DECAY_RATE = 0.005  # Exponential decay rate for exploration prob

RewardHistory = []

for episode in range(TOTAL_EPISODES):
    state = env.reset()
    step = 0
    done = False
    EpisodeReward = 0

    for step in range(MAX_STEPS):
        # Epsilon-greedy algorithm: choose best action with probabiliy 1 -
        # epsilon. Choose random action (i.e. explore) with probability epsilon.
        # In the beginning, epsilon == 1, so we're in all exploration mode.
        # Gradually we reduce epsilon and take more educated actions from the
        # qtable.
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

    # Reduce epsilon (because we need less and less exploration)
    epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
        np.exp(-DECAY_RATE * episode)
    RewardHistory.append(EpisodeReward)

print(qtable)

# p = figure(sizing_mode='stretch_both')
# p.circle(range(len(RewardHistory)), RewardHistory,
#          size=1, color="navy", alpha=0.5)
# show(p)

# Run the trained agent on a few games
for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("Ep: {}".format(episode))

    for step in range(MAX_STEPS):
        action = np.argmax(qtable[state, :])
        nState, reward, done, info = env.step(action)

        if done:
            env.render()
            print("Number of steps: {}".format(step))
            break

        state = nState

env.close()
