import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

"""
At any point we are dealing with only one bandit, represented by self.state ==
the index of that bandit in self.bandits.
"""


class ContextualBandit():
    def __init__(self):
        self.state = 0  # index of the bandit, 0, 1, 2.
        self.bandits = np.array([
            [0.2, 0, -0.0, -5],  # Bandit 0: arm 3 most optimal
            [0.1, -5, 1, 0.25],  # Bandit 1: arm 1 most optimal
            [-5, 5.0, 5.0, 5.],  # Bandit 2: arm 0 most optimal
        ])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        """
        Randomly set the bandit index in the state.
        """
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pullArm(self, action):
        """
        Pull an arm of the bandit given by state (bandit index) and action
        (bandit arm index), and get reward for action.
        """
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1  # Positive reward.
        else:
            return -1  # Negative reward.


class agent():
    def __init__(self, lr, s_size, a_size):
        """
        lr: learning rate
        a_size: size of action collection
        s_size: size of bandit collection
        """
        # These lines established the feed-forward part of the network. The
        # agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_OH,
                                      a_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # The next six lines establish the training proceedure. We feed the
        # reward and chosen action into the network to compute the loss, and use
        # it to update the network.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        self.update = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(self.loss)


tf.reset_default_graph()

cBandit = ContextualBandit()
myAgent = agent(lr=0.001, s_size=cBandit.num_bandits,
                a_size=cBandit.num_actions)

weights = tf.trainable_variables()[0]

total_episodes = 10000
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])
epsilon = 0.1  # Chance of taking a random action (epsilon-greedy algorithm)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    i = 0
    while i < total_episodes:
        s = cBandit.getBandit()

        if np.random.rand(1) < epsilon:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(myAgent.chosen_action,
                              feed_dict={myAgent.state_in: [s]})

        reward = cBandit.pullArm(action)
        _, ww = sess.run([myAgent.update, weights],
                         feed_dict={
                             myAgent.reward_holder: [reward],
                             myAgent.action_holder: [action],
                             myAgent.state_in: [s]})
        total_reward[s, action] += reward

        if i % 500 == 0:
            print("Bandit rewards:\n {}".format(total_reward))

        i += 1

for a in range(cBandit.num_bandits):
    print("Action {} for bandit {} is most optimal: {}".format(
        np.argmax(ww[a]),
        a,
        np.argmax(ww[a]) == np.argmin(cBandit.bandits[a])))
