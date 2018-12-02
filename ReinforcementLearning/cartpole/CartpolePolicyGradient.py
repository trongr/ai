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

            # Placeholder so we can look at this in tensorboard
            self.MeanReward = tf.placeholder(tf.float32, name="MeanReward")

            fc1 = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=10,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            fc2 = tf.contrib.layers.fully_connected(
                inputs=fc1,
                num_outputs=ACTION_SIZE,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            fc3 = tf.contrib.layers.fully_connected(
                inputs=fc2,
                num_outputs=ACTION_SIZE,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.actionDistr = tf.nn.softmax(fc3)

            """
            tf.nn.softmax_cross_entropy_with_logits computes the cross entropy
            of the result after applying the softmax function. If you have
            single-class labels, where an object can only belong to one class,
            you might now consider using
            tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't
            have to convert your labels to a dense one-hot array.
            """
            negLogProb = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=fc3, labels=actions)
            self.loss = loss = tf.reduce_mean(negLogProb * discountedRewards)
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.minimize = optimizer.minimize(loss)

    def save(self, SAVE_DIR_WITH_PREFIX, episode):
        savepath = Saver.save(
            self.sess, SAVE_DIR_WITH_PREFIX, global_step=episode)
        print("Save path: {}".format(savepath))

    @staticmethod
    def makeActionOneHot(action, ACTION_SIZE):
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
    def discountRewards(EpisodeRewards):
        """
        Discount and normalize rewards
        """
        discountedRewards = np.zeros_like(EpisodeRewards)
        cumulative = 0.0
        for i in reversed(range(len(EpisodeRewards))):
            cumulative = cumulative * GAMMA + EpisodeRewards[i]
            discountedRewards[i] = cumulative

        mean = np.mean(discountedRewards)
        std = np.std(discountedRewards)
        discountedRewards = (discountedRewards - mean) / (std)

        return discountedRewards


env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

STATE_SIZE = 4
ACTION_SIZE = env.action_space.n
MAX_EPISODES = 10000
LEARNING_RATE = 0.01
GAMMA = 0.95

tf.reset_default_graph()
sess = GetTFSession()
agent = Agent(sess, STATE_SIZE, ACTION_SIZE)
sess.run(tf.global_variables_initializer())

# NOTE. Make sure this folder exists
SAVE_DIR = "./models/"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

"""
Launch tensorboard with:
tensorboard --logdir=./logs/
"""
writer = tf.summary.FileWriter("./logs/")
tf.summary.scalar("Loss", agent.loss)
tf.summary.scalar("MeanReward", agent.MeanReward)
writeOp = tf.summary.merge_all()

AllRewards = []
TotalReward = 0
MaxReward = 0
episode = 0

for episode in range(MAX_EPISODES):
    EpisodeStates, EpisodeActions, EpisodeRewards = [], [], []
    EpisodeRewardsSum = 0
    state = env.reset()
    # env.render()

    while True:
        actionDistr = sess.run(agent.actionDistr, feed_dict={
            agent.inputs: state.reshape([1, 4])})
        action = np.random.choice(range(actionDistr.shape[1]),
                                  p=actionDistr.ravel())

        nstate, reward, done, info = env.step(action)

        EpisodeStates.append(state)
        action = Agent.makeActionOneHot(action, ACTION_SIZE)
        EpisodeActions.append(action)
        EpisodeRewards.append(reward)

        if done:
            EpisodeRewardsSum = np.sum(EpisodeRewards)
            AllRewards.append(EpisodeRewardsSum)
            TotalReward = np.sum(AllRewards)
            MeanReward = np.divide(TotalReward, episode + 1)
            MaxReward = np.amax(AllRewards)

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", EpisodeRewardsSum)
            print("Mean Reward", MeanReward)
            print("Max Reward: ", MaxReward)

            discountedRewards = Agent.discountRewards(EpisodeRewards)

            loss_, _ = sess.run([agent.loss, agent.minimize], feed_dict={
                agent.inputs: np.vstack(np.array(EpisodeStates)),
                agent.actions: np.vstack(np.array(EpisodeActions)),
                agent.discountedRewards: discountedRewards})

            summary = sess.run(writeOp, feed_dict={
                agent.inputs: np.vstack(np.array(EpisodeStates)),
                agent.actions: np.vstack(np.array(EpisodeActions)),
                agent.discountedRewards: discountedRewards,
                agent.MeanReward: MeanReward})
            writer.add_summary(summary, episode)
            writer.flush()

            EpisodeStates, EpisodeActions, EpisodeRewards = [], [], []

            break

        state = nstate
