"""
Doom Gathering Health with Policy Gradient.
"""

import glob
from collections import deque
import vizdoom
import numpy as np
from skimage import transform
import tensorflow as tf


def GetTFSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


class Agent:
    def __init__(self, sess, STATE_SIZE, ACTION_SIZE, name="Agent"):
        self.sess = sess
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *STATE_SIZE], name="inputs")
            self.actions = tf.placeholder(tf.int32, [None, ACTION_SIZE], name="actions")
            self.discountedRewards = tf.placeholder(
                tf.float32, [None], name="discountedRewards"
            )

            flat1 = tf.layers.flatten(self.inputs)

            fc1 = tf.layers.dense(
                inputs=flat1,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1",
            )

            logits = tf.layers.dense(
                inputs=fc1,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=ACTION_SIZE,
                activation=None,
            )

            self.actionDistr = tf.nn.softmax(logits)

            self.xentropy = xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=self.actions
            )
            self.loss = tf.reduce_mean(xentropy * self.discountedRewards)

            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
            self.minimize = optimizer.minimize(self.loss)

    def save(self, Saver, SAVE_PATH_PREFIX, ep):
        """ Save model. """
        savepath = Saver.save(self.sess, SAVE_PATH_PREFIX, global_step=ep)
        print("Save path: {}".format(savepath))

    def chooseAction(self, state):
        """
        Run the state on the agent's self.actionDistr and return an action. Also
        implements epsilon greedy algorithm: if random > epsilon, we choose the
        "best" action from the network, otw we choose an action randomly.
        """
        actionDistr = sess.run(
            self.actionDistr, feed_dict={self.inputs: state.reshape(1, *STATE_SIZE)}
        )
        action = np.random.choice(range(ACTION_SIZE), p=actionDistr.ravel())
        return ACTIONS[action]

    # Remove the conv layers. It might be too powerful causing the network to
    # overfit
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

        maxReward = np.max(discountedRewards)
        discountedRewards = discountedRewards - maxReward

        # L = len(discountedRewards)
        # for i in range(L):
        #     flippity = (i + 1.0) / L
        #     if random.uniform(0, 1) > flippity:
        #         discountedRewards[i] *= -1.0

        print(
            "Discounted rewards {}".format(
                np.concatenate((discountedRewards[:20], discountedRewards[-20:]))
            )
        )

        return discountedRewards


def makeEnv():
    """ Create the environment """
    game = vizdoom.DoomGame()
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
    return deque(
        [np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int) for i in range(length)],
        maxlen=length,
    )


game, ACTIONS = makeEnv()

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
STATE_SIZE = [FRAME_WIDTH, FRAME_HEIGHT]
ACTION_SIZE = game.get_available_buttons_size()
LEARNING_RATE = 0.002  # ALPHA
GAMMA = 0.8  # Discounting rate
MAX_EPS = 500

tf.reset_default_graph()
sess = GetTFSession()
agent = Agent(sess, STATE_SIZE, ACTION_SIZE)
sess.run(tf.global_variables_initializer())

SAVE_DIR = "./save/"
Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
if glob.glob(SAVE_DIR + "/*"):
    Saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

"""
Launch tensorboard with: tensorboard --logdir=./logs/
"""
writer = tf.summary.FileWriter("./logs/")
tf.summary.scalar("Loss", agent.loss)
SummaryOp = tf.summary.merge_all()

"""
Minibatch data. Episode states, actions, and rewards data get concatenated into
these guys, which are fed into the network every say 10 episodes.
"""
statesMB, actionsMB, discountedRewardsMB = [], [], []

for ep in range(MAX_EPS):
    states, actions, rewards = [], [], []
    game.new_episode()
    step = 0
    done = False

    while not done:
        step += 1

        frame = PreprocFrame(game.get_state().screen_buffer)
        action = agent.chooseAction(frame)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        states.append(frame)
        actions.append(action)
        rewards.append(reward)

        if done:
            statesMB = statesMB + states
            actionsMB = actionsMB + actions
            nrewards = Agent.discountNormalizeRewards(rewards)
            discountedRewardsMB = discountedRewardsMB + nrewards.tolist()

        if done and ep % 10 == 0:
            xentropy, loss, _, summary = sess.run(
                [agent.xentropy, agent.loss, agent.minimize, SummaryOp],
                feed_dict={
                    agent.inputs: np.array(statesMB),
                    agent.actions: np.array(actionsMB),
                    agent.discountedRewards: discountedRewardsMB,
                },
            )

            print("========================================")
            print("Ep: {} / {}".format(ep, MAX_EPS))
            print("Loss: {}".format(loss))
            print("Steps: {}".format(step))
            print(
                "Cross Entropy: {}".format(
                    np.concatenate((xentropy[:10], xentropy[-10:]))
                )
            )

            writer.add_summary(summary, ep)
            writer.flush()

    if ep % 10 == 0:
        agent.save(Saver, SAVE_DIR + "/save", ep)
