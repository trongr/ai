import gym
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

import matplotlib

# For some reason you need this to fix matplotlib crashing on OSX.
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

"""
TODO. Replace Cartpole with this Pong environment.
"""


class AtariPreprocessedEnv:
    """ A warper for atari game environment.
    To use this warpper, replace 'env = gym.make("CartPole-v1")' with
    'env = AtariPreprocessedEnv()' and paste the definition of this warpper
    before that statement (class should be defined before use)
    """

    def __init__(self, name="PongDeterministic-v0"):
        self.env = gym.make(name)
        self.action_space = self.env.action_space
        self.observation_space = (80 * 80,)

        self.pre_image = None

    def _preprocess(self, img):
        """ transform a 210x160x3 uint8 image to a 6400x1 float vector
        Crop, down-sample, erase background and set foreground to 1
        ref: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
        """
        img = img[35:195]
        img = img[::2, ::2, 0]
        img[img == 144] = 0
        img[img == 109] = 0
        img[img != 0] = 1
        curr = img.astype(np.float).ravel()
        # Subtract the last preprocessed image.
        diff = (
            curr - self.pre_image if self.pre_image is not None else np.zeros_like(curr)
        )
        self.pre_image = curr
        return diff

    def reset(self):
        self.pre_image = None
        return self._preprocess(self.env.reset())

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self._preprocess(o), r, d, i

    def render(self):
        self.env.render()


class Episode:
    """
    Keeps track of observations, actions, and rewards for each step in an
    episode.
    """

    def __init__(self):
        self.observation = []
        self.action = []
        self.reward = []

    def append(self, s, a, r):
        self.observation.append(s)
        self.action.append(a)
        self.reward.append(r)


class Agent:
    """
    Agent network.
    """

    def __init__(self, action_space, ctx):
        """
        Define the network, a multilayer perceptron (MLP).
        """
        self.action_number = action_space.n

        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(200, activation="relu"))
            net.add(gluon.nn.Dense(self.action_number))

        net.collect_params().initialize(mx.init.Xavier(magnitude=0.5), ctx=ctx)
        self.trainer = gluon.Trainer(
            net.collect_params(), "adam", {"learning_rate": 1e-3}
        )

        self.net = net
        self.ctx = ctx

    def act(self, observation):
        """
        Choose an action based on agent's current network states.
        """
        # transform a single observation to a batch, necessary for shape inference
        observation = [observation]
        obs = mx.nd.array(observation, ctx=self.ctx)

        policy = self.net(obs)[0]  # get the first item of return batch
        policy = mx.nd.softmax(policy).asnumpy()

        return np.random.choice(np.arange(self.action_number), p=policy)

    def train(self, episode):
        """
        Train agent on observations, actions, and rewards from a single episode.
        """
        # State, action, and reward
        s, a, r = episode.observation, episode.action, episode.reward

        GAMMA = 0.99  # Discounting factor
        cumulative = 0
        for i in reversed(range(len(r))):
            cumulative = cumulative * GAMMA + r[i]
            r[i] = cumulative

        s = mx.nd.array(s, ctx=self.ctx)
        a = mx.nd.array(a, ctx=self.ctx).one_hot(self.action_number)
        r = mx.nd.array(r, ctx=self.ctx).reshape((len(r), 1))

        # substract baseline
        r -= mx.nd.mean(r)

        # forward
        with autograd.record():
            output = self.net(s)
            policy = mx.nd.softmax(output)
            policy = mx.nd.clip(policy, 1e-6, 1 - 1e-6)  # clip for log
            log_policy = mx.nd.log(policy)
            loss = -mx.nd.mean(r * log_policy * a)

        loss.backward()
        self.trainer.step(len(s))


def run_episode(env, agent, render):
    """
    Run agent through a single episode. Choose action based on agent's current
    state, without actually training it. Training is done after this episode is
    over.
    """
    episode = Episode()
    observation = env.reset()
    done = False

    while not done:
        action = agent.act(observation)
        next_observation, reward, done, _ = env.step(action)
        episode.append(observation, action, reward)
        observation = next_observation

        if render:
            env.render()

    return episode


env = gym.make("CartPole-v1")
agent = Agent(env.action_space, mx.cpu())

print("observation space : {}".format(env.observation_space))
print("action space : {}".format(env.action_space))

n_episodes = 1000
mean_reward = 0  # running reward average
reward_log = []

RENDER = True

"""
Run agent through n_episodes. For each, collect the reward in a running average.
At the same time train the agent on the episode.
"""

for i in range(n_episodes):
    episode = run_episode(env, agent, render=RENDER)
    reward = sum(episode.reward)

    # save moving average of episode reward for plotting
    if i == 0:
        mean_reward = reward
    else:
        mean_reward = 0.99 * mean_reward + 0.01 * reward

    reward_log.append(mean_reward)

    agent.train(episode)

    if i % 50 == 0:
        print("episode: %d  reward: %.2f  mean reward: %.2f" % (i, reward, mean_reward))

plt.plot(reward_log)
plt.legend(["mean reward"])
plt.show()
