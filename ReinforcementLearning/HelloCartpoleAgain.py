import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


def getDiscountedReward(reward):
    """
    Take 1D float array of rewards and compute discounted reward
    """
    gamma = 0.99
    discountedReward = np.zeros_like(reward)
    total = 0
    for t in reversed(range(0, reward.size)):
        total = total * gamma + reward[t]
        discountedReward[t] = total
    return discountedReward


class CartpoleAgent():
    """
    The agent is trained to take input and produce an action.
    """

    def __init__(self, lr, stateSize, actionSize, hiddenSize):
        """
        - lr: Learning rate.

        - self.output is the action distribution, i.e. a list of probabilities
          associated with each action.
        """
        # Feed-forward. The agent takes a state and produces an action prob
        # distribution
        self.state = state = tf.placeholder(
            shape=[None, stateSize], dtype=tf.float32)
        hidden = slim.fully_connected(
            state, hiddenSize, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = output = slim.fully_connected(
            hidden, actionSize, activation_fn=tf.nn.softmax, biases_initializer=None)

        # The next six lines establish the training proceedure. We feed the
        # reward and chosen action into the network to compute the loss, and use
        # it to update the network.
        self.rewardHolder = rewardHolder = tf.placeholder(
            shape=[None], dtype=tf.float32)
        self.actionHolder = actionHolder = tf.placeholder(
            shape=[None], dtype=tf.int32)

        indexes = tf.range(0, tf.shape(output)[
                           0]) * tf.shape(output)[1] + actionHolder
        responsibleOutputs = tf.gather(tf.reshape(output, [-1]), indexes)
        loss = - tf.reduce_mean(tf.log(responsibleOutputs) * rewardHolder)

        tvars = tf.trainable_variables()
        self.gradientHolders = gradientHolders = [
            tf.placeholder(tf.float32, name=str(idx) + '_holder')
            for idx, var in enumerate(tvars)]

        self.gradients = tf.gradients(loss, tvars)

        self.updateBatch = tf.train.AdamOptimizer(learning_rate=lr).apply_gradients(
            zip(gradientHolders, tvars))

    @staticmethod
    def chooseAction(actionDist):
        """
        Probabilistically pick an action given action prob dist (self.output).
        """
        actionProb = np.random.choice(actionDist[0], p=actionDist[0])
        action = np.argmax(actionDist == actionProb)
        return action


tf.reset_default_graph()

agent = CartpoleAgent(lr=1e-2, stateSize=4, actionSize=2, hiddenSize=8)

# An experiment is one run of Cartpole balancing, e.g. ends when the stick
# falls over. I'm guessing. Is this true??
TOTAL_EPISODES = 5000  # Total number of experiments to run.
MAX_EP_TIME = 999  # Total number of timesteps per experiment / episode.
UPDATE_FREQUENCY = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    TotalReward = []
    i = 0
    while i < TOTAL_EPISODES:
        state = env.reset()
        RunningReward = 0
        epHist = []
        for j in range(MAX_EP_TIME):
            # Feed state to agent and get action
            actionDist = sess.run(agent.output, feed_dict={
                                  agent.state: [state]})
            action = CartpoleAgent.chooseAction(actionDist)

            # Apply agent action to environment
            nState, reward, done, _ = env.step(action)
            epHist.append([state, action, reward, nState])
            state = nState
            RunningReward += reward

            # Update the agent network when the experiment is over
            if done == True:
                epHist = np.array(epHist)
                epHist[:, 2] = getDiscountedReward(epHist[:, 2])

                grads = sess.run(agent.gradients, feed_dict={
                                 agent.rewardHolder: epHist[:, 2],
                                 agent.actionHolder: epHist[:, 1],
                                 agent.state: np.vstack(epHist[:, 0])})

                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                # Why do we only update the agent after 5 episodes? It's only
                # using the last 5 gradients to update the agent?
                if i % UPDATE_FREQUENCY == 0 and i != 0:
                    sess.run(agent.updateBatch, feed_dict=dict(
                        zip(agent.gradientHolders, gradBuffer)))
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                TotalReward.append(RunningReward)
                break

        if i % 100 == 0:
            print("Mean rewards: {}".format(np.mean(TotalReward[-100:])))

        i += 1
