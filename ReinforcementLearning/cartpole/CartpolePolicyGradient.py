import tensorflow as tf
import numpy as np
import gym
import glob


def GetTFSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class Agent:
    def __init__(self, sess, STATE_SIZE, ACTION_SIZE, name='Agent'):
        self.sess = sess
        with tf.variable_scope(name):
            self.inputs = inputs = tf.placeholder(
                tf.float32, [None, STATE_SIZE], name="inputs")
            self.actions = actions = tf.placeholder(
                tf.int32, [None, ACTION_SIZE], name="actions")
            self.discountedRewards = discountedRewards = tf.placeholder(
                tf.float32, [None, ], name="discountedRewards")

            fc1 = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=10,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=10,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            fc3 = tf.contrib.layers.fully_connected(
                inputs=fc2,
                num_outputs=ACTION_SIZE,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.actionDistr = tf.nn.softmax(fc3)

            negLogProb = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=fc3, labels=actions)
            self.loss = loss = tf.reduce_mean(negLogProb * discountedRewards)
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.minimize = optimizer.minimize(loss)

    def save(self, Saver, SAVE_DIR_WITH_PREFIX, episode):
        """
        Save model.
        """
        savepath = Saver.save(
            self.sess, SAVE_DIR_WITH_PREFIX, global_step=episode)
        print("Save path: {}".format(savepath))

    @staticmethod
    def onehotAction(action, ACTION_SIZE):
        """
        PARAMS
        - action: index of the action.
        - ACTION_SIZE: number of possible actions

        Returns action as one-hot vector.
        """
        onehot = np.zeros(ACTION_SIZE)
        onehot[action] = 1
        return onehot

    @staticmethod
    def discountNormalizeRewards(EpisodeRewards):
        """
        PARAMS
        - EpisodeRewards: list of rewards for current episode

        Returns discounted and normalized rewards
        """
        discountedRewards = np.zeros_like(EpisodeRewards)
        cumulative = 0.0
        for i in reversed(range(len(EpisodeRewards))):
            cumulative = cumulative * GAMMA + EpisodeRewards[i]
            discountedRewards[i] = cumulative

        mean = np.mean(discountedRewards)
        std = np.std(discountedRewards)
        discountedRewards = (discountedRewards - mean) / std

        return discountedRewards


env = gym.make('CartPole-v0')
env = env.unwrapped
env.reset()
env.seed(1)

STATE_SIZE = 4
ACTION_SIZE = env.action_space.n
MAX_EPISODES = 100000
LEARNING_RATE = 0.01
GAMMA = 0.95

tf.reset_default_graph()
sess = GetTFSession()
agent = Agent(sess, STATE_SIZE, ACTION_SIZE)
sess.run(tf.global_variables_initializer())

SAVE_DIR = "./save/"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

"""
Launch tensorboard with:
tensorboard --logdir=./logs/
"""
writer = tf.summary.FileWriter("./logs/")
tf.summary.scalar("Loss", agent.loss)
writeOp = tf.summary.merge_all()

episode = 0

for episode in range(MAX_EPISODES):
    EpisodeStates, EpisodeActions, EpisodeRewards = [], [], []
    state = env.reset()
    TotalReward = 0
    done = False

    while not done:
        actionDistr = sess.run(agent.actionDistr, feed_dict={
            agent.inputs: state.reshape([1, 4])})
        action = np.random.choice(range(actionDistr.shape[1]),
                                  p=actionDistr.ravel())

        nstate, reward, done, info = env.step(action)

        env.render()

        EpisodeStates.append(state)
        action = Agent.onehotAction(action, ACTION_SIZE)
        EpisodeActions.append(action)
        EpisodeRewards.append(reward)
        TotalReward += reward

        if done:
            discountedRewards = Agent.discountNormalizeRewards(EpisodeRewards)

            summary, loss, _ = sess.run([writeOp, agent.loss, agent.minimize], feed_dict={
                agent.inputs: np.vstack(np.array(EpisodeStates)),
                agent.actions: np.vstack(np.array(EpisodeActions)),
                agent.discountedRewards: discountedRewards})

            # TotalReward is also the number of steps in the episode. Each extra
            # step gives 1 point reward.
            print("==========================================")
            print("Episode:", episode)
            print("Reward:", TotalReward)
            print("Loss:", loss)

            writer.add_summary(summary, episode)
            writer.flush()

        state = nstate

    if episode % 50 == 0:
        agent.save(Saver, SAVE_DIR + "/save", episode)

env.close()
