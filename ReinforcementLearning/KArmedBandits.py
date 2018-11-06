import tensorflow as tf
import numpy as np

# List out our bandits. Currently bandit 4 (index#3) is set to most often
# provide a positive reward.
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)


def pullBandit(bandit):
    """
    The smaller the bandit, the more likely it is to give a positive reward.
    """
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1


def eGreedyAction(epsilon, num_bandits, sess, chosen_action):
    """
    Choose random action with probability epsilon, choose from network with prob 1 -
    epsilon
    """
    if np.random.rand(1) < epsilon:
        action = np.random.randint(num_bandits)
    else:
        action = sess.run(chosen_action)
    return action


tf.reset_default_graph()

# These two lines established the feed-forward part of the network. This
# does the actual choosing.
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

# The next six lines establish the training proceedure. We feed the reward and
# chosen action into the network to compute the loss, and use it to update the
# network.
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)  # Updates all variables, e.g. weights.

total_episodes = 1000  # Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits)  # Set scoreboard for bandits to 0.
epsilon = 0.1  # Set the chance of taking a random action.

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    i = 0
    while i < total_episodes:
        action = eGreedyAction(epsilon, num_bandits, sess, chosen_action)
        reward = pullBandit(bandits[action])
        total_reward[action] += reward

        _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={
            reward_holder: [reward], action_holder: [action]})

        if i % 50 == 0:
            print("{} Running bandit: Rewards: {}. Responsible weights: {}".format(
                i, total_reward, ww))

        i += 1

print("The agent thinks bandit {} is the most promising....".format(np.argmax(ww) + 1))
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
