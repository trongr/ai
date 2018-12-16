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
        with tf.variable_scope(name):
            self.inputs = inputs = tf.placeholder(
                tf.float32, [None, *STATE_SIZE], name="inputs")
            self.actions = actions = tf.placeholder(
                tf.int32, [None, ACTION_SIZE], name="actions")
            self.discountedRewards = discountedRewards = tf.placeholder(
                tf.float32, [None, ], name="discountedRewards")

            """
            First convnet: CNN BatchNormalization ELU
            """

            # Input is 84x84x4
            conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            bn1 = tf.layers.batch_normalization(
                conv1, training=True, epsilon=1e-5, name='bn1')

            elu1 = tf.nn.elu(bn1, name="conv1_out")  # --> [20, 20, 32]

            """
            Second convnet: CNN BatchNormalization ELU
            """

            conv2 = tf.layers.conv2d(
                inputs=elu1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            bn2 = tf.layers.batch_normalization(
                conv2, training=True, epsilon=1e-5, name='bn2')

            elu2 = tf.nn.elu(bn2, name="elu2")  # --> [9, 9, 64]

            """
            Third convnet: CNN BatchNormalization ELU
            """

            conv3 = tf.layers.conv2d(
                inputs=elu2,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            bn3 = tf.layers.batch_normalization(
                conv3, training=True, epsilon=1e-5, name='bn3')

            elu3 = tf.nn.elu(bn3, name="elu3")  # --> [3, 3, 128]

            flatten = tf.layers.flatten(elu3)  # --> [1152]

            fc4 = tf.layers.dense(
                inputs=flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc4")

            logits = tf.layers.dense(
                inputs=fc4,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=ACTION_SIZE,
                activation=None)

            self.actionDistr = tf.nn.softmax(logits)

            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=actions)
            self.loss = tf.reduce_mean(xentropy * self.discountedRewards)

            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
            self.train_opt = optimizer.minimize(self.loss)

    def save(self, Saver, SAVE_PATH_PREFIX, ep):
        """ Save model. """
        savepath = Saver.save(self.sess, SAVE_PATH_PREFIX, global_step=ep)
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
            cumulative = EpisodeRewards[i] + GAMMA * cumulative
            discountedRewards[i] = cumulative

        mean = np.mean(discountedRewards)
        std = np.std(discountedRewards)
        discountedRewards = (discountedRewards - mean) / std

        return discountedRewards


def makeEnv():
    """
    Create the environment
    """
    game = DoomGame()
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    game.init()
    ACTIONS = np.identity(3, dtype=int).tolist()  # [[1,0,0], [0,1,0], [0,0,1]]
    return game, ACTIONS


def PreprocFrame(frame):
    """
    Preprocess the frame: crop, normalize, and resize.
    """
    # Remove the roof because it contains no information.
    cropped = frame[80:, :]
    normalized = cropped / 255.0
    frame = transform.resize(normalized, [84, 84])
    return frame


def InitDeque(length):
    """
    Initialize deque with zero-images, one array for each image
    """
    return deque([np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int)
                  for i in range(length)], maxlen=length)


def StackFrames(frames, state, isNewEpisode):
    """
    Stack frames together to give network a sense of time. If this is a new
    episode, duplicate the frame over the stacked frames. OTW add this new frame
    to the existing queue.

    @param {*} state: A single frame from the game.

    @return {*} nstate, frames. nstate is the stacked frames.
    """
    frame = PreprocFrame(state)
    if isNewEpisode:
        frames = InitDeque(NUM_FRAMES)  # Clear frames
        # Because we're in a new episode, copy the same frame 4x
        frames.append(frame)
        frames.append(frame)
        frames.append(frame)
        frames.append(frame)
        nstate = np.stack(frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        frames.append(frame)
        nstate = np.stack(frames, axis=2)
    return nstate, frames


game, ACTIONS = makeEnv()

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
NUM_FRAMES = 4  # Stack 4 frames together
STATE_SIZE = [FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES]
ACTION_SIZE = game.get_available_buttons_size()
LEARNING_RATE = 0.002  # ALPHA
GAMMA = 0.95  # Discounting rate
MAX_EPOCHS = 500
BATCH_SIZE = 3000

frames = InitDeque(NUM_FRAMES)  # The stacked frames

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
SummaryOp = tf.summary.merge_all()

"""
TRAINING / TESTING
"""

TRAINING = True  # Set to False to test trained agent on games
if TRAINING:
    epoch = 0
    while epoch < MAX_EPOCHS:
        states = []
        actions = []
        epRewards = []
        batchRewards = []
        discountedRewards = []

        game.new_episode()
        state = game.get_state().screen_buffer
        state, frames = StackFrames(frames, state, True)

        ep = 0
        step = 0  # Step in an episode

        while True:
            actionDistr = sess.run(agent.actionDistr, feed_dict={
                agent.inputs: state.reshape(1, *STATE_SIZE)})
            action = np.random.choice(range(actionDistr.shape[1]),
                                      p=actionDistr.ravel())
            action = ACTIONS[action]

            reward = game.make_action(action)
            done = game.is_episode_finished()
            step += 1

            states.append(state)
            actions.append(action)
            epRewards.append(reward)

            if done:
                nstate = np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int)
                nstate, frames = StackFrames(frames, nstate, False)

                batchRewards.append(epRewards)
                discountedRewards.append(
                    Agent.discountNormalizeRewards(epRewards))

                print("Steps: {}".format(step))
                step = 0

                # If the number of batchRewards > BATCH_SIZE stop the minibatch
                # creation (Because we have sufficient number of episode mb).
                # Remember that we put this condition here, because we want entire
                # episode (Monte Carlo) so we can't check that condition for each
                # step but only if an episode is finished
                if len(np.concatenate(batchRewards)) > BATCH_SIZE:
                    break

                epRewards = []
                ep += 1

                game.new_episode()
                state = game.get_state().screen_buffer
                state, frames = StackFrames(frames, state, True)
            else:
                nstate = game.get_state().screen_buffer
                nstate, frames = StackFrames(frames, nstate, False)
                state = nstate

        statesMB = np.stack(np.array(states))
        actionsMB = np.stack(np.array(actions))
        batchRewards = np.concatenate(batchRewards)
        discountedRewardsMB = np.concatenate(discountedRewards)

        summary, loss, _ = sess.run([
            SummaryOp, agent.loss, agent.train_opt], feed_dict={
            agent.inputs: statesMB.reshape((-1, *STATE_SIZE)),
            agent.actions: actionsMB,
            agent.discountedRewards: discountedRewardsMB})

        print("============================================================")
        print("Epoch: {} / {}".format(epoch, MAX_EPOCHS))
        print("Num Eps: {}".format(ep))
        print("Training Loss: {}".format(loss))

        writer.add_summary(summary, epoch)
        writer.flush()

        if epoch % 10 == 0:
            agent.save(Saver, SAVE_DIR + "/save", epoch)

        epoch += 1
