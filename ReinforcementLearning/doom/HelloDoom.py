
import tensorflow as tf
import numpy as np
from vizdoom import *
import random
from skimage import transform
from collections import deque
import glob


def GetTFSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class DQNetwork:
    """
    The agent
    """

    def __init__(self, sess, STATE_SIZE, ACTION_SIZE, LEARNING_RATE,
                 name='DQNetwork'):
        self.sess = sess
        with tf.variable_scope(name):
            FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES = STATE_SIZE
            self.inputs = inputs = tf.placeholder(
                tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES], name="inputs")
            self.actions = actions = tf.placeholder(
                tf.float32, [None, ACTION_SIZE], name="actions")
            self.targetQ = targetQ = tf.placeholder(
                tf.float32, [None], name="target")

            """
            First convnet: CNN > BatchNormalization > ELU
            """

            # Input shape [BATCH_SIZE, 84, 84, 4]
            conv1 = tf.layers.conv2d(
                inputs=inputs,
                filters=32, kernel_size=[8, 8],
                strides=[4, 4], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")
            # output shape [BATCH_SIZE, 20, 20, 32]

            batchnorm1 = tf.layers.batch_normalization(
                conv1, training=True, epsilon=1e-5, name='batchnorm1')
            # output shape [BATCH_SIZE, 20, 20, 32]

            elu1 = tf.nn.elu(batchnorm1, name="elu1")
            # output shape [BATCH_SIZE, 20, 20, 32]

            """
            Second convnet: CNN > BatchNormalization > ELU
            """

            conv2 = tf.layers.conv2d(
                inputs=elu1, filters=64,
                kernel_size=[4, 4], strides=[2, 2], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")
            # output shape [BATCH_SIZE, 9, 9, 64]. TODO@trong If you use SAME
            # padding as opposed to VALID, output should have the same width and
            # height as input, e.g. this one would be (BATCH_SIZE, 20, 20, 64).
            # Double check. You'll need to down sample if I recall.

            batchnorm2 = tf.layers.batch_normalization(
                conv2, training=True, epsilon=1e-5, name='batchnorm2')
            # output shape [BATCH_SIZE, 9, 9, 64]

            elu2 = tf.nn.elu(batchnorm2, name="elu2")
            # output shape [BATCH_SIZE, 9, 9, 64]

            """
            Third convnet: CNN > BatchNormalization > ELU
            """

            conv3 = tf.layers.conv2d(
                inputs=elu2, filters=128, kernel_size=[4, 4],
                strides=[2, 2], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")
            # output shape [BATCH_SIZE, 3, 3, 128]

            batchnorm3 = tf.layers.batch_normalization(
                conv3, training=True, epsilon=1e-5, name='batchnorm3')
            # output shape [BATCH_SIZE, 3, 3, 128]

            elu3 = tf.nn.elu(batchnorm3, name="elu3")
            # output shape [BATCH_SIZE, 3, 3, 128]

            """
            Final FC layers
            """

            # Might need this one for diff tensorflow versions.
            # flatten4 = tf.layers.flatten(elu3)
            flatten4 = tf.contrib.layers.flatten(elu3)
            # output shape [BATCH_SIZE, 1152]

            fc5 = tf.layers.dense(
                inputs=flatten4,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")  # output shape [BATCH_SIZE, 512]

            self.output = output = tf.layers.dense(
                inputs=fc5,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)  # output shape [BATCH_SIZE, 3]

            """
            Training
            """

            predictedQ = tf.reduce_sum(output * actions, axis=1)
            self.loss = loss = tf.reduce_mean(tf.square(targetQ - predictedQ))
            self.optimizer = tf.train.RMSPropOptimizer(
                LEARNING_RATE).minimize(loss)

    def save(self, SAVE_DIR_WITH_PREFIX, episode):
        savepath = Saver.save(
            self.sess, SAVE_DIR_WITH_PREFIX, global_step=episode)
        print("Save path: {}".format(savepath))


def CreateGameEnv():
    game = DoomGame()
    game.load_config("DoomBasicConfig.cfg")
    game.set_doom_scenario_path("DoomBasicData.wad")
    game.init()
    game.new_episode()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    PossibleActions = [left, right, shoot]

    return game, PossibleActions


def PreprocessFrame(frame):
    """
    Take a frame, crop the roof because it contains no useful information.
    Resize, normalize, and return preprocessedFrame. We don't grayscale the
    frame because it's already done in the vizdoom config.
    """
    croppedFrame = frame[30:-10, 30:-30]
    normalizedFrame = croppedFrame / 255.0
    preprocessedFrame = transform.resize(normalizedFrame, [84, 84])
    return preprocessedFrame


def InitDeque(length):
    """
    Initialize deque with zero-images, one array for each image
    """
    return deque([np.zeros((84, 84), dtype=np.int)
                  for i in range(length)], maxlen=length)


def StackFrames(StackedFrames, state, isNewEpisode):
    """
    Stack frames together to give network a sense of time. If this is a new
    episode, duplicate the frame over the StackedState. OTW add this new frame
    to the existing queue.

    @param {*} state: A single frame from the game.

    @return {*} StackedState, StackedFrames
    """
    frame = PreprocessFrame(state)

    if isNewEpisode:
        StackedFrames = InitDeque(NUM_FRAMES)  # Clear StackedFrames

        # Because we're in a new episode, copy the same frame 4x
        StackedFrames.append(frame)
        StackedFrames.append(frame)
        StackedFrames.append(frame)
        StackedFrames.append(frame)

        StackedState = np.stack(StackedFrames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        StackedFrames.append(frame)
        StackedState = np.stack(StackedFrames, axis=2)

    return StackedState, StackedFrames


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


def PredictAction(agent, DecayStep, state, PossibleActions):
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
        FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES = state.shape
        Qs = sess.run(agent.output, feed_dict={agent.inputs: state.reshape(
            (1, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES))})

        choice = np.argmax(Qs)
        action = PossibleActions[int(choice)]

    return action, ExploreProbability


game, PossibleActions = CreateGameEnv()

"""
MODEL PARAMETERS
"""

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
NUM_FRAMES = 4  # Stack 4 frames together
STATE_SIZE = [FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES]
ACTION_SIZE = game.get_available_buttons_size()  # 3 actions: left, right, shoot
LEARNING_RATE = 0.0002  # alpha
GAMMA = 0.95  # Discounting rate

StackedFrames = InitDeque(NUM_FRAMES)

"""
TRAINING HYPERPARAMETERS
"""

NUM_TRAIN_EPISODES = 500
NUM_TEST_EPISODES = 100
MAX_STEPS = 100  # Max possible steps in an episode
BATCH_SIZE = 64

# Exploration parameters for epsilon greedy strategy
DecayStep = 0

tf.reset_default_graph()
sess = GetTFSession()

# The network isn't quite the agent. The agent consists of the network plus
# epsilon greedy. But that's OK; no need to make such minute distinctions.
agent = DQNetwork(sess, STATE_SIZE, ACTION_SIZE, LEARNING_RATE)
sess.run(tf.global_variables_initializer())

# NOTE. Make sure this folder exists
SAVE_DIR = "./models/HelloDoom"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

"""
Launch tensorboard with:
tensorboard --logdir=./tmp/HelloDoomTensorBoard/
"""
writer = tf.summary.FileWriter("./tmp/HelloDoomTensorBoard/")
tf.summary.scalar("Loss", agent.loss)
writeOp = tf.summary.merge_all()

"""
Training / testing
"""

TRAINING = False  # Set to False to test trained agent on games

if TRAINING == True:  # TRAIN AGENT
    memory = Memory(MEMORY_SIZE=1000000)

    """
    Pretraining. Before training, we want to fill the replay buffer / memory
    with experiences, so that later we can sample from it in the training loop
    below.
    """

    for i in range(BATCH_SIZE):
        if i == 0:
            state = game.get_state().screen_buffer
            state, StackedFrames = StackFrames(StackedFrames, state, True)

        action = random.choice(PossibleActions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        # Episode is done when you kill the monster, or when time runs out at
        # 300 steps.
        if done:
            nstate = np.zeros(state.shape)
            memory.add((state, action, reward, nstate, done))
            game.new_episode()
            state = game.get_state().screen_buffer
            state, StackedFrames = StackFrames(StackedFrames, state, True)
        else:
            nstate = game.get_state().screen_buffer
            nstate, StackedFrames = StackFrames(StackedFrames, nstate, False)
            memory.add((state, action, reward, nstate, done))
            state = nstate

    for episode in range(NUM_TRAIN_EPISODES):
        step = 0
        EpisodeRewards = []

        game.new_episode()
        state = game.get_state().screen_buffer
        state, StackedFrames = StackFrames(StackedFrames, state, True)
        done = False

        while not done and step < MAX_STEPS:
            step += 1
            DecayStep += 1

            """
            Gathering experience into the replay buffer.
            """

            action, ExploreProbability = PredictAction(
                agent, DecayStep, state, PossibleActions)

            reward = game.make_action(action)
            done = game.is_episode_finished()
            EpisodeRewards.append(reward)

            if done:
                nstate = np.zeros((84, 84), dtype=np.int)
                nstate, StackedFrames = StackFrames(
                    StackedFrames, nstate, False)
                TotalRewards = np.sum(EpisodeRewards)
                memory.add((state, action, reward, nstate, done))

                print("Episode: {}. ".format(episode) +
                      "TotalReward: {}. ".format(TotalRewards) +
                      "Loss: {:.4f}. ".format(loss) +
                      "ExploreProbability: {:.4f}".format(ExploreProbability))
            else:
                nstate = game.get_state().screen_buffer
                nstate, StackedFrames = StackFrames(
                    StackedFrames, nstate, False)
                memory.add((state, action, reward, nstate, done))
                state = nstate

            """
            Independently learning from the replay buffer instead of the
            experiences we just saw.
            """

            batch = memory.sample(BATCH_SIZE)
            statesMB = np.array([each[0] for each in batch], ndmin=3)
            actionsMB = np.array([each[1] for each in batch])
            rewardsMB = np.array([each[2] for each in batch])
            nStatesMB = np.array([each[3] for each in batch], ndmin=3)
            doneMB = np.array([each[4] for each in batch])

            targetQsBatch = []

            Qs = sess.run(agent.output, feed_dict={
                agent.inputs: nStatesMB})

            # Set Q_target = r if the episode ends at s+1, otherwise set
            # Q_target = r + GAMMA * max_{a'} Q(s', a').
            for i in range(0, len(batch)):
                isDone = doneMB[i]
                if isDone:
                    targetQsBatch.append(rewardsMB[i])
                else:
                    target = rewardsMB[i] + GAMMA * np.max(Qs[i])
                    targetQsBatch.append(target)

            targetsMB = np.array([each for each in targetQsBatch])

            loss, _ = sess.run([agent.loss, agent.optimizer], feed_dict={
                               agent.inputs: statesMB,
                               agent.targetQ: targetsMB,
                               agent.actions: actionsMB})

            """
            Saving training statistics
            """

            summary = sess.run(writeOp, feed_dict={
                               agent.inputs: statesMB,
                               agent.targetQ: targetsMB,
                               agent.actions: actionsMB})
            writer.add_summary(summary, episode)
            writer.flush()

        if episode % 10 == 0:
            agent.save(SAVE_DIR + "/save", episode)

else:  # TEST AGENT
    game, PossibleActions = CreateGameEnv()

    for episode in range(NUM_TEST_EPISODES):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, StackedFrames = StackFrames(StackedFrames, state, True)

        step = 0
        done = False
        while not done:
            step += 1

            Qs = sess.run(agent.output, feed_dict={
                          agent.inputs: state.reshape(
                              (1, FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES))})

            choice = np.argmax(Qs)
            action = PossibleActions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()

            if done:
                break
            else:
                nstate = game.get_state().screen_buffer
                nstate, StackedFrames = StackFrames(
                    StackedFrames, nstate, False)
                state = nstate

        score = game.get_total_reward()
        print("Episode: {}. Steps taken: {}. Score: {}".format(episode, step, score))

    game.close()
