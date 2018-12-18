import tensorflow as tf
import numpy as np
from vizdoom import *
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import glob


def GetTFSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class Agent:

    def __init__(self, sess, STATE_SIZE, ACTION_SIZE, name='Agent'):
        self.sess = sess
        self.ExploreExploitProb = 1.0
        with tf.variable_scope(name):
            self.inputs = inputs = tf.placeholder(
                tf.float32, [None, *STATE_SIZE], name="inputs")
            self.actions = actions = tf.placeholder(
                tf.int32, [None, ACTION_SIZE], name="actions")
            self.discountedRewards = discountedRewards = tf.placeholder(
                tf.float32, [None, ], name="discountedRewards")

            """ First convnet: CNN BatchNormalization ELU """

            # Input is 84x84x4
            conv1 = tf.layers.conv2d(
                inputs=inputs,
                filters=32,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="SAME",
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                name="conv1")

            bn1 = tf.layers.batch_normalization(
                conv1, training=True, epsilon=1e-5, name='bn1')

            elu1 = tf.nn.elu(bn1, name="conv1_out")

            """ Second convnet: CNN BatchNormalization ELU """

            conv2 = tf.layers.conv2d(
                inputs=elu1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="SAME",
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                name="conv2")

            bn2 = tf.layers.batch_normalization(
                conv2, training=True, epsilon=1e-5, name='bn2')

            elu2 = tf.nn.elu(bn2, name="elu2")

            # """ Third convnet: CNN BatchNormalization ELU """

            # conv3 = tf.layers.conv2d(
            #     inputs=elu2,
            #     filters=128,
            #     kernel_size=[4, 4],
            #     strides=[2, 2],
            #     padding="SAME",
            #     use_bias=True,
            #     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #     bias_initializer=tf.zeros_initializer(),
            #     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            #     bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
            #     name="conv3")

            # bn3 = tf.layers.batch_normalization(
            #     conv3, training=True, epsilon=1e-5, name='bn3')

            # elu3 = tf.nn.elu(bn3, name="elu3")

            """ Final FC layers and softmax """

            # TODO. Remember to turn this on if you turn conv3 layers back on:
            # flat4 = tf.layers.flatten(elu3)

            flat4 = tf.layers.flatten(elu2)

            fc4 = tf.layers.dense(
                inputs=flat4,
                units=512,
                activation=tf.nn.elu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                name="fc4")

            fc5 = tf.layers.dense(
                inputs=fc4,
                units=256,
                activation=tf.nn.elu,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                name="fc5")

            self.logits = logits = tf.layers.dense(
                inputs=fc5,
                units=ACTION_SIZE,
                activation=None,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                name="logits")

            self.actionDistr = tf.nn.softmax(logits)

            self.xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=actions)
            self.loss = tf.reduce_mean(self.xentropy * discountedRewards) + \
                tf.losses.get_regularization_loss()

            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.minimize = optimizer.minimize(self.loss)

    def chooseAction(self, state):
        """
        Run the state on the agent's self.actionDistr and return an action. Also
        implements epsilon greedy algorithm: if random > epsilon, we choose the
        "best" action from the network, otw we choose an action randomly.
        """
        # poij Turn this on after implementing memory
        # In the beginning, epsilon == 1.0, so this is all exploration. As
        # training goes on, we reduce epsilon every episode.
        # if random.uniform(0, 1) > self.ExploreExploitProb:  # EXPLOIT
        #     actionDistr, logits = sess.run([
        #         self.actionDistr, self.logits], feed_dict={
        #         self.inputs: state.reshape(1, *STATE_SIZE)})
        #     action = np.random.choice(range(ACTION_SIZE), p=actionDistr.ravel())
        # else:  # EXPLORE RANDOMLY
        #     action = np.random.choice(range(ACTION_SIZE))
        actionDistr, logits = sess.run([
            self.actionDistr, self.logits], feed_dict={
            self.inputs: state.reshape(1, *STATE_SIZE)})
        action = np.random.choice(range(ACTION_SIZE), p=actionDistr.ravel())
        return ACTIONS[action]

    def updateExploreExploitEpsilon(self, episode):
        """
        This method should be called every episode to reduce epsilon and change
        the explore/exploit ratio.
        """
        MIN_EPSILON = 0.01
        MAX_EPSILON = 1.0
        DECAY_RATE = 0.005
        self.ExploreExploitProb = (MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) *
                                   np.exp(-DECAY_RATE * episode))
        print("Explore exploit prob: {}".format(self.ExploreExploitProb))

    def save(self, Saver, SAVE_PATH_PREFIX, ep):
        """ Save model. """
        savepath = Saver.save(self.sess, SAVE_PATH_PREFIX, global_step=ep)
        print("Save path: {}".format(savepath))

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
            cumulative = EpisodeRewards[i] + GAMMA * cumulative
            discountedRewards[i] = cumulative

        mean = np.mean(discountedRewards)
        std = np.std(discountedRewards)
        discountedRewards = (discountedRewards - mean) / std

        return discountedRewards


def makeEnv():
    """ Create the environment """
    game = DoomGame()
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    game.init()
    ACTIONS = np.identity(3, dtype=int).tolist()  # [[1,0,0], [0,1,0], [0,0,1]]
    return game, ACTIONS


def PreprocFrame(frame):
    """ Preprocess the frame: crop, normalize, and resize. """
    # Remove the roof because it contains no information.
    cropped = frame[80:, :]
    normalized = cropped / 255.0
    frame = transform.resize(normalized, [84, 84])
    return frame


def InitDeque(length):
    """ Initialize deque with zero-images, one array for each image """
    return deque([np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int)
                  for i in range(length)], maxlen=length)


def StackFrames(frames, frame):
    """
    Stack frames together to give network a sense of time. If this is a new
    episode, duplicate the frame over the stacked frames. OTW add this new frame
    to the existing queue.

    @param {*} frame: A single frame from the game.

    @return {*} state, frames. state is the stacked frames.
    """
    frame = PreprocFrame(frame)
    if frames is None:
        # We're in a new episode: initialize queue store consecutive frames
        frames = InitDeque(NUM_FRAMES)
        frames.append(frame)
        frames.append(frame)
        frames.append(frame)
        frames.append(frame)
        state = np.stack(frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        frames.append(frame)
        state = np.stack(frames, axis=2)
    return state, frames


game, ACTIONS = makeEnv()

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
NUM_FRAMES = 4  # Stack 4 frames together
STATE_SIZE = [FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES]
ACTION_SIZE = game.get_available_buttons_size()
LEARNING_RATE = 0.002  # ALPHA
GAMMA = 0.95  # Discounting rate
MAX_EPS = 10000

tf.reset_default_graph()
sess = GetTFSession()
agent = Agent(sess, STATE_SIZE, ACTION_SIZE)
sess.run(tf.global_variables_initializer())

SAVE_DIR = "./save/"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

""" Launch tensorboard with: tensorboard --logdir=./logs/ """
writer = tf.summary.FileWriter("./logs/")
tf.summary.scalar("Loss", agent.loss)
SummaryOp = tf.summary.merge_all()

for ep in range(MAX_EPS):
    states, actions, rewards = [], [], []
    game.new_episode()
    frames = None
    step = 0  # Step in an episode
    done = False

    while not done:
        step += 1

        frame = game.get_state().screen_buffer
        state, frames = StackFrames(frames, frame)
        action = agent.chooseAction(state)

        reward = game.make_action(action)
        done = game.is_episode_finished()

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if done:
            agent.updateExploreExploitEpsilon(ep)
            discountedRewards = Agent.discountNormalizeRewards(rewards)
            summary, loss, _, xentropy = sess.run([
                SummaryOp, agent.loss, agent.minimize, agent.xentropy], feed_dict={
                agent.inputs: np.array(states),
                agent.actions: np.array(actions),
                agent.discountedRewards: discountedRewards})

            print("========================================")
            print("Ep: {} / {}".format(ep, MAX_EPS))
            print("Loss: {}".format(loss))
            print("Steps: {}".format(step))
            print("poij xentropy", xentropy)

            writer.add_summary(summary, ep)
            writer.flush()

    if ep % 10 == 0:
        agent.save(Saver, SAVE_DIR + "/save", ep)
