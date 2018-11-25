"""
REQUIRES
================================================================================
    - python 3.7 for OpenAI retro. Run `conda activate py37`
"""

import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import random
import glob


def GetTFSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class Memory():
    """
    Experience / replay buffer
    """

    def __init__(self, MEMORY_SIZE=1000000):
        self.buffer = deque(maxlen=MEMORY_SIZE)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        bufsize = len(self.buffer)
        index = np.random.choice(
            np.arange(bufsize), size=BATCH_SIZE, replace=False)
        return [self.buffer[i] for i in index]


class DQNetwork:
    """
    The agent
    """

    def __init__(self, sess, STATE_SIZE, ACTION_SIZE, LEARNING_RATE, name='DQNetwork'):
        self.sess = sess
        self.STATE_SIZE = STATE_SIZE
        self.ACTION_SIZE = ACTION_SIZE
        self.LEARNING_RATE = LEARNING_RATE

        with tf.variable_scope(name):
            self.inputs = inputs = tf.placeholder(
                tf.float32, [None, *STATE_SIZE], name="inputs")
            self.actions = actions = tf.placeholder(
                tf.float32, [None, ACTION_SIZE], name="actions")
            self.targetQ = targetQ = tf.placeholder(
                tf.float32, [None], name="target")

            """
            First convnet: CNN > BatchNormalization > ELU
            """

            # TODO@trong Figure out the layer dimensions.

            # Input shape [BATCH_SIZE, 110, 84, 4]
            conv1 = tf.layers.conv2d(
                inputs=inputs,
                filters=32, kernel_size=[4, 4],
                strides=[2, 2], padding="SAME",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")
            # batchnorm1 = tf.layers.batch_normalization(
            #     conv1, training=True, epsilon=1e-5, name='batchnorm1')
            # elu1 = tf.nn.elu(batchnorm1, name="elu1")
            """Get rid of batchnorm to reduce compute"""
            elu1 = tf.nn.elu(conv1, name="elu1")
            # output shape ??

            """
            Second convnet: CNN > BatchNormalization > ELU
            """

            conv2 = tf.layers.conv2d(
                inputs=elu1, filters=64,
                kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")
            # output shape ??
            # batchnorm2 = tf.layers.batch_normalization(
            #     conv2, training=True, epsilon=1e-5, name='batchnorm2')
            # elu2 = tf.nn.elu(batchnorm2, name="elu2")
            elu2 = tf.nn.elu(conv2, name="elu2")
            # output shape ??

            """
            Third convnet: CNN > BatchNormalization > ELU
            """

            conv3 = tf.layers.conv2d(
                inputs=elu2, filters=128, kernel_size=[4, 4],
                strides=[2, 2], padding="SAME",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")
            # batchnorm3 = tf.layers.batch_normalization(
            #     conv3, training=True, epsilon=1e-5, name='batchnorm3')
            # elu3 = tf.nn.elu(batchnorm3, name="elu3")
            elu3 = tf.nn.elu(conv3, name="elu3")
            # output shape ??

            """
            Final FC layers
            """

            # Might need this one for diff tensorflow versions.
            # flatten4 = tf.layers.flatten(elu3)
            flatten4 = tf.contrib.layers.flatten(elu3)
            # output shape ??

            fc5 = tf.layers.dense(
                inputs=flatten4,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")  # output shape [BATCH_SIZE, 512]

            self.output = output = tf.layers.dense(
                inputs=fc5,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=ACTION_SIZE,
                activation=None)  # output shape [BATCH_SIZE, 3]

            """
            Training
            """

            predictedQ = tf.reduce_sum(output * actions, axis=1)
            self.loss = loss = tf.reduce_mean(tf.square(targetQ - predictedQ))
            self.optimizer = tf.train.RMSPropOptimizer(
                LEARNING_RATE).minimize(loss)

    def PredictAction(self, DecayStep, state, PossibleActions):
        """
        Choose action using epsilon greedy: choose random action with
        ExploreProbability (EXPLORE), OTW choose best action from network (EXPLOIT).
        """
        # ExploreProbability starts at 1, and decays exponentially with rate
        # DECAY_RATE and approaches EXPLORE_STOP.
        EXPLORE_START = 1.0  # Exploration probability at start
        EXPLORE_STOP = 0.01  # Minimum exploration probability
        DECAY_RATE = 0.0001  # Exponential decay rate for exploration prob

        ExploreProbability = EXPLORE_STOP + \
            (EXPLORE_START - EXPLORE_STOP) * np.exp(-DECAY_RATE * DecayStep)

        if (ExploreProbability > np.random.rand()):  # EXPLORE
            action = random.choice(PossibleActions)
        else:  # EXPLOIT
            Qs = sess.run(self.output, feed_dict={self.inputs: state.reshape(
                (1, *state.shape))})
            choice = np.argmax(Qs)
            action = PossibleActions[int(choice)]

        return action, ExploreProbability


def PreprocessFrame(frame):
    """
    PreprocessFrame: Grayscale, resize, normalize. Return preprocessed_frame
    """
    gray = rgb2gray(frame)
    croppedFrame = gray[8:-12, 4:-12]  # Remove the part below the player
    normalizedFrame = croppedFrame / 255.0
    preprocessedFrame = transform.resize(normalizedFrame, [110, 84])
    return preprocessedFrame  # 110x84x1 frame


def InitDeque(length):
    """
    Initialize deque with zero-images, one array for each image
    """
    return deque([np.zeros((110, 84), dtype=np.int)
                  for i in range(length)], maxlen=length)


def StackFrames(StackedFrames, state, is_new_episode):
    frame = PreprocessFrame(state)

    if is_new_episode:
        StackedFrames = InitDeque(NUM_FRAMES)

        # Because we're in a new episode, copy the same frame 4x
        StackedFrames.append(frame)
        StackedFrames.append(frame)
        StackedFrames.append(frame)
        StackedFrames.append(frame)

        # Stack the frames
        StackedState = np.stack(StackedFrames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        StackedFrames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        StackedState = np.stack(StackedFrames, axis=2)

    return StackedState, StackedFrames


env = retro.make(game='SpaceInvaders-Atari2600')

"""
MODEL PARAMETERS
"""

FRAME_WIDTH = 110
FRAME_HEIGHT = 84
NUM_FRAMES = 4  # Stack 4 frames together
STATE_SIZE = [FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES]
ACTION_SIZE = env.action_space.n  # 8 possible actions
LEARNING_RATE = 0.0002
GAMMA = 0.9

"""
TRAINING HYPERPARAMETERS
"""

NUM_TRAIN_EPISODES = 500
NUM_TEST_EPISODES = 100
MAX_STEPS = 50000
BATCH_SIZE = 64

DecayStep = 0
StackedFrames = InitDeque(NUM_FRAMES)
PossibleActions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

tf.reset_default_graph()
sess = GetTFSession()
agent = DQNetwork(sess, STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
sess.run(tf.global_variables_initializer())

# NOTE. Make sure this folder exists
SAVE_DIR = "./models/SpaceInvaders"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

"""
Launch tensorboard with:
tensorboard --logdir=./tensorboard/
"""
writer = tf.summary.FileWriter("./tensorboard/")
tf.summary.scalar("Loss", agent.loss)
writeOp = tf.summary.merge_all()

"""
Training / testing
"""

TRAINING = True
RENDER_ENV = True

if TRAINING == True:  # TRAIN AGENT
    memory = Memory(MEMORY_SIZE=1000000)

    """
    Pretraining. Before training, we want to fill the replay buffer / memory
    with experiences, so that later we can sample from it in the training loop
    below.
    """

    for i in range(BATCH_SIZE):
        if i == 0:
            state = env.reset()
            state, StackedFrames = StackFrames(StackedFrames, state, True)

        action = random.choice(PossibleActions)
        nstate, reward, done, _ = env.step(action)

        if RENDER_ENV:
            env.render()

        if done:  # Episode is done when you die 3 times
            nstate = np.zeros(state.shape)
            memory.add((state, action, reward, nstate, done))
            state = env.reset()
            state, StackedFrames = StackFrames(StackedFrames, state, True)
        else:
            nstate, StackedFrames = StackFrames(StackedFrames, nstate, False)
            memory.add((state, action, reward, nstate, done))
            state = nstate

    for episode in range(NUM_TRAIN_EPISODES):
        step = 0
        EpisodeReward = 0
        state = env.reset()
        state, StackedFrames = StackFrames(StackedFrames, state, True)
        done = False

        while not done and step < MAX_STEPS:
            step += 1
            DecayStep += 1

            """
            Gathering experience into the replay buffer.
            """

            action, ExploreProbability = agent.PredictAction(
                DecayStep, state, PossibleActions)
            nstate, reward, done, _ = env.step(action)
            EpisodeReward += reward

            if RENDER_ENV:
                env.render()

            if done:
                nstate = np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int)
                nstate, StackedFrames = StackFrames(
                    StackedFrames, nstate, False)
                memory.add((state, action, reward, nstate, done))
                print("Episode: {}. ".format(episode) +
                      "EpisodeReward: {}. ".format(EpisodeReward) +
                      "Loss: {:.4f}. ".format(loss) +
                      "ExploreProbability: {:.4f}".format(ExploreProbability))
            else:
                nstate, StackedFrames = StackFrames(
                    StackedFrames, nstate, False)
                memory.add((state, action, reward, nstate, done))
                state = nstate

            """
            Independently learning from the replay buffer instead of the
            experiences we just saw.
            """

            batch = memory.sample(BATCH_SIZE)
            statesMemBuf = np.array([each[0] for each in batch], ndmin=3)
            actionsMemBuf = np.array([each[1] for each in batch])
            rewardsMemBuf = np.array([each[2] for each in batch])
            nStatesMemBuf = np.array([each[3] for each in batch], ndmin=3)
            doneMemBuf = np.array([each[4] for each in batch])

            Qs = sess.run(agent.output, feed_dict={agent.inputs: nStatesMemBuf})

            targetQsBatch = []
            for i in range(0, len(batch)):
                isDone = doneMemBuf[i]
                if isDone:
                    targetQsBatch.append(rewardsMemBuf[i])
                else:
                    target = rewardsMemBuf[i] + GAMMA * np.max(Qs[i])
                    targetQsBatch.append(target)

            targetsMemBuf = np.array([each for each in targetQsBatch])

            loss, _ = sess.run([agent.loss, agent.optimizer], feed_dict={
                               agent.inputs: statesMemBuf,
                               agent.targetQ: targetsMemBuf,
                               agent.actions: actionsMemBuf})

            """
            Saving training statistics
            """

            summary = sess.run(writeOp, feed_dict={
                               agent.inputs: statesMemBuf,
                               agent.targetQ: targetsMemBuf,
                               agent.actions: actionsMemBuf})
            writer.add_summary(summary, episode)
            writer.flush()

        if episode % 10 == 0:
            agent.save(SAVE_DIR + "/save", episode)
else:  # TEST AGENT
    for episode in range(NUM_TEST_EPISODES):
        EpisodeReward = 0
        state = env.reset()
        state, StackedFrames = StackFrames(StackedFrames, state, True)
        step = 0
        done = False
        while not done:
            step += 1
            Qs = sess.run(agent.output, feed_dict={
                          agent.inputs: state.reshape((1, *STATE_SIZE))})
            choice = np.argmax(Qs)
            action = PossibleActions[int(choice)]
            nstate, reward, done, _ = env.step(action)
            EpisodeReward += reward

            if RENDER_ENV:
                env.render()

            if done:
                break
            else:
                nstate, StackedFrames = StackFrames(
                    StackedFrames, nstate, False)
                state = nstate

        print("Episode: {}. Steps taken: {}. EpisodeReward: {}".format(
            episode, step, EpisodeReward))

env.close()
