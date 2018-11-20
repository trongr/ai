
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


class DQNetwork:
    def __init__(self, sess, state_size, action_size, learning_rate,
                 name='DQNetwork'):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            frameWidth, frameHeight, numFrames = state_size
            self.inputs = tf.placeholder(
                tf.float32, [None, frameWidth, frameHeight, numFrames], name="inputs")
            self.actions = tf.placeholder(
                tf.float32, [None, self.action_size], name="actions")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet: CNN > BatchNormalization > ELU
            """

            # Input shape [84, 84, 4]
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32, kernel_size=[8, 8],
                strides=[4, 4], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")
            # output shape??

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1, training=True, epsilon=1e-5, name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # output shape [20, 20, 32]

            """
            Second convnet: CNN > BatchNormalization > ELU
            """

            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out, filters=64,
                kernel_size=[4, 4], strides=[2, 2], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")
            # shape?? TODO@trong. If you use SAME padding as opposed to VALID,
            # output should have the same width and height as input, e.g. this
            # one would be (20, 20, 64). Double check. You'll need to down
            # sample if I recall.

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2, training=True, epsilon=1e-5, name='batch_norm2')
            # output shape same as previous layer

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # output shape [9, 9, 64]

            """
            Third convnet: CNN > BatchNormalization > ELU
            """

            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out, filters=128, kernel_size=[4, 4],
                strides=[2, 2], padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")
            # shape??

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3, training=True, epsilon=1e-5, name='batch_norm3')
            # output shape same as previous layer

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # output shape [3, 3, 128]

            # self.flatten = tf.layers.flatten(self.conv3_out)
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # output shape [1152]

            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")  # output shape [512]

            self.output = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)  # output shape [3]

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(self.output * self.actions, axis=1)

            # The loss is the difference between our predicted Q_values and the
            # Q_target Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)

    def save(self, SAVE_DIR_WITH_PREFIX, episode):
        savepath = Saver.save(
            self.sess, SAVE_DIR_WITH_PREFIX, global_step=episode)
        print("Save path: {}".format(savepath))


def createGameEnv():
    game = DoomGame()
    game.load_config("DoomBasicConfig.cfg")
    game.set_doom_scenario_path("DoomBasicData.wad")
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    PossibleActions = [left, right, shoot]

    return game, PossibleActions


def preprocess_frame(frame):
    """
    Take a frame, crop the roof because it contains no useful information.
    Resize it. Normalize it. Return preprocessed_frame. We don't grayscale the
    frame because it's already done in the vizdoom config.
    """
    cropped_frame = frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


def initDeque(length):
    """
    Initialize deque with zero-images one array for each image
    """
    return deque([np.zeros((84, 84), dtype=np.int)
                  for i in range(length)], maxlen=length)


def stack_frames(stacked_frames, state, is_new_episode):
    """
    Stack frames together to give network a sense of time.
    """
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = initDeque(stack_size)  # Clear stacked_frames

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


def predict_action(agent, explore_start, explore_stop, decay_rate,
                   decay_step, state, PossibleActions):
    """
    Choose action using epsilon greedy: choose random action with
    explore_probability (EXPLORE), OTW choose best action from network
    (EXPLOIT).
    """
    # TODO@trong What are these explore_start/stop options?
    # TODO@trong Remember explore_probability from previous session.
    explore_probability = explore_stop + \
        (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > np.random.rand()):  # EXPLORE
        action = random.choice(PossibleActions)
    else:  # EXPLOIT
        frameWidth, frameHeight, numFrames = state.shape
        Qs = sess.run(agent.output, feed_dict={agent.inputs: state.reshape(
            (1, frameWidth, frameHeight, numFrames))})

        choice = np.argmax(Qs)
        action = PossibleActions[int(choice)]

    return action, explore_probability


game, PossibleActions = createGameEnv()
game.new_episode()

"""
MODEL PARAMETERS
"""

FRAME_WIDTH = 84
stack_size = 4  # Stack 4 frames
stacked_frames = initDeque(stack_size)
state_size = [FRAME_WIDTH, FRAME_WIDTH, stack_size]
action_size = game.get_available_buttons_size()  # 3 actions: left, right, shoot
learning_rate = 0.0002  # alpha
gamma = 0.95  # Discounting rate

"""
TRAINING HYPERPARAMETERS
"""

total_episodes = 500  # Total episodes for training
max_steps = 100  # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob
decay_step = 0

"""
MEMORY PARAMETERS
"""

# Number of experiences stored in the Memory when initialized for the first time
PRETRAIN_LENGTH = batch_size
memory_size = 1000000  # Number of experiences the Memory can keep
memory = Memory(max_size=memory_size)

# TODO@trong What does pretrain do?
for i in range(PRETRAIN_LENGTH):
    if i == 0:
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    action = random.choice(PossibleActions)
    reward = game.make_action(action)
    done = game.is_episode_finished()

    # done is True when episode is over. Might be different depending on
    # scenario??
    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(
            stacked_frames, next_state, False)
        memory.add((state, action, reward, next_state, done))
        state = next_state

tf.reset_default_graph()
sess = GetTFSession()

# The network isn't quite the agent. The agent consists of the network plus
# epsilon greedy. But that's OK; no need to make such minute distinctions.
agent = DQNetwork(sess, state_size, action_size, learning_rate)
sess.run(tf.global_variables_initializer())

# NOTE. Make sure this folder exists
SAVE_DIR = "./models/HelloDoom"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

"""
TODO@trong Watch this:
Launch tensorboard with:
tensorboard --logdir=./tmp/HelloDoomTensorBoard/
"""
writer = tf.summary.FileWriter("./tmp/HelloDoomTensorBoard/")
tf.summary.scalar("Loss", agent.loss)
writeOp = tf.summary.merge_all()

"""
SCRIPT PARAMETERS
"""
TRAINING = True  # Set to False to test trained agent on games
EPISODE_RENDER = False  # I think it always renders

if TRAINING == True:  # TRAIN AGENT
    for episode in range(total_episodes):
        step = 0
        episode_rewards = []

        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        while not done and step < max_steps:
            step += 1
            decay_step += 1

            """
            Gathering experience into the replay buffer / Memory.
            """

            action, explore_probability = predict_action(
                agent, explore_start, explore_stop, decay_rate,
                decay_step, state, PossibleActions)

            reward = game.make_action(action)
            done = game.is_episode_finished()
            episode_rewards.append(reward)

            if done:
                next_state = np.zeros((84, 84), dtype=np.int)
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                total_reward = np.sum(episode_rewards)
                memory.add((state, action, reward, next_state, done))

                print("Episode: {}. Total reward: {}. Loss: {:.4f}. Explore Prob: {:.4f}".format(
                    episode, total_reward, loss, explore_probability))
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                memory.add((state, action, reward, next_state, done))
                state = next_state

            """
            Learning
            """

            batch = memory.sample(batch_size)
            statesMemBuf = np.array([each[0] for each in batch], ndmin=3)
            actionsMemBuf = np.array([each[1] for each in batch])
            rewardsMemBuf = np.array([each[2] for each in batch])
            nStatesMemBuf = np.array([each[3] for each in batch], ndmin=3)
            doneMemBuf = np.array([each[4] for each in batch])

            target_Qs_batch = []

            Qs_next_state = sess.run(agent.output, feed_dict={
                agent.inputs: nStatesMemBuf})

            # Set Q_target = r if the episode ends at s+1, otherwise set
            # Q_target = r + gamma * max_{a'} Q(s', a').
            for i in range(0, len(batch)):
                isDone = doneMemBuf[i]
                if isDone:
                    target_Qs_batch.append(rewardsMemBuf[i])
                else:
                    target = rewardsMemBuf[i] + gamma * \
                        np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targetsMemBuf = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([agent.loss, agent.optimizer], feed_dict={
                agent.inputs: statesMemBuf,
                agent.target_Q: targetsMemBuf,
                agent.actions: actionsMemBuf})

            summary = sess.run(writeOp, feed_dict={
                agent.inputs: statesMemBuf,
                agent.target_Q: targetsMemBuf,
                agent.actions: actionsMemBuf})
            writer.add_summary(summary, episode)
            writer.flush()

        if episode % 10 == 0:
            agent.save(SAVE_DIR + "/save", episode)

else:  # TEST AGENT
    game, PossibleActions = createGameEnv()
    totalScore = 0

    for i in range(1):
        done = False
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        frameWidth, frameHeight, numFrames = state.shape

        while not game.is_episode_finished():
            Qs = sess.run(DQNetwork.output, feed_dict={
                DQNetwork.inputs: state.reshape((1, frameWidth, frameHeight, numFrames))})

            choice = np.argmax(Qs)
            action = PossibleActions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()

            if done:
                break
            else:
                print("The game continues")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(
                    stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: {}".format(score))

    game.close()
