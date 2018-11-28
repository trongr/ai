import tensorflow as tf
import numpy as np
from vizdoom import *
import random
from skimage import transform
from collections import deque
import glob


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6]

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        """
        Remember that our tree is composed of a sum tree that contains the
        priority scores at his leaf And also a data array We don't use deque
        because it means that at each timestep our experiences change index by
        one. We prefer to use a simple array and to overwrite when the memory is
        full.
        """
        self.tree = SumTree(capacity)

    def store(self, experience):
        """
        Store a new experience in our tree. Each new experience have a score of
        max_prority (it will be then improved when we use this exp to train our
        DDQN)
        """
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will
        # never have a chance to be selected So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max p for new p

    def sample(self, n):
        """
        - First, to sample a minibatch of k size, the range [0, priority_total]
          is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score
          correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        """

        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty(
            (n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n
        # ranges
        priority_segment = self.tree.total_priority / n       # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min(
            [1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.maximum(1, np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority)
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(
                n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, treeIdx, abs_errors):
        """
        Update the priorities on the tree
        """
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(treeIdx, ps):
            self.tree.update(ti, p)


class DDDQNet:
    """
    Network for the Agent and the TargetNetwork.
    """

    def __init__(self, sess, STATE_SIZE, ACTION_SIZE, LEARNING_RATE, name):
        self.sess = sess
        # We use tf.variable_scope here to know which network we're using (agent
        # or TargetNetwork), e.g. when we update our w parameters (by copying
        # the DQN parameters)
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, *STATE_SIZE], name="inputs")
            self.ISWeights = tf.placeholder(
                tf.float32, [None, 1], name='ISWeights')
            self.actions = tf.placeholder(
                tf.float32, [None, ACTION_SIZE], name="actions")
            self.targetQ = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet: CNN ELU
            """
            # Input is 100x120x4
            conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")
            conv1_out = tf.nn.elu(conv1, name="conv1_out")

            """
            Second convnet: CNN ELU
            """

            conv2 = tf.layers.conv2d(
                inputs=conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")
            conv2_out = tf.nn.elu(conv2, name="conv2_out")

            """
            Third convnet: CNN ELU
            """

            conv3 = tf.layers.conv2d(
                inputs=conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")
            conv3_out = tf.nn.elu(conv3, name="conv3_out")

            flatten = tf.layers.flatten(conv3_out)

            """
            Dueling DQN
            """

            # Here we separate into two streams: state value V(s)
            value_fc = tf.layers.dense(
                inputs=flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value_fc")
            value = tf.layers.dense(
                inputs=value_fc,
                units=1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value")

            # Action value A(s, a)
            advantage_fc = tf.layers.dense(
                inputs=flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantage_fc")
            advantage = tf.layers.dense(
                inputs=advantage_fc,
                units=ACTION_SIZE,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantages")

            # Aggregating layer merges the two streams back: Q(s,a) = V(s) +
            # A(s, a) - 1/|A| * sum A(s, a'). Why subtract the mean advantage?
            self.output = value + advantage - tf.reduce_mean(
                advantage, axis=1, keepdims=True)

            Q = tf.reduce_sum(tf.multiply(
                self.output, self.actions), axis=1)

            # The loss is modified because of PER
            self.AbsoluteErrors = tf.abs(self.targetQ - Q)
            self.loss = tf.reduce_mean(
                self.ISWeights * tf.squared_difference(self.targetQ, Q))
            self.optimizer = tf.train.RMSPropOptimizer(
                LEARNING_RATE).minimize(self.loss)

    def save(self, SAVE_DIR_WITH_PREFIX, episode):
        savepath = Saver.save(
            self.sess, SAVE_DIR_WITH_PREFIX, global_step=episode)
        print("Save path: {}".format(savepath))

    def PredictAction(self, DecayStep, state, ACTIONS):
        """
        Choose action using epsilon greedy: choose random action with
        ExploreProb (EXPLORE), OTW choose best action from network (EXPLOIT).
        """
        # ExploreProb starts at 1, and decays exponentially with rate DECAY_RATE
        # and approaches EXPLORE_STOP.
        EXPLORE_START = 1.0  # Exploration probability at start
        EXPLORE_STOP = 0.01  # Minimum exploration probability
        DECAY_RATE = 0.0001  # Exponential decay rate for exploration prob

        ExploreProb = EXPLORE_STOP + \
            (EXPLORE_START - EXPLORE_STOP) * np.exp(-DECAY_RATE * DecayStep)

        if (ExploreProb > np.random.rand()):  # EXPLORE
            action = random.choice(ACTIONS)
        else:  # EXPLOIT
            Qs = sess.run(self.output, feed_dict={
                self.inputs: state.reshape((1, *state.shape))})
            choice = np.argmax(Qs)
            action = ACTIONS[int(choice)]

        return action, ExploreProb

    def updateTargetGraph(self):
        """
        This function helps us to copy one set of variables to another. In our case
        we use it when we want to copy the parameters of DQN to Target_network.
        Thanks of the very good implementation of Arthur Juliani
        https://github.com/awjuliani
        """
        fromVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "agent")
        toVars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
        opHolder = [toVar.assign(fromVar)
                    for fromVar, toVar in zip(fromVars, toVars)]
        return opHolder


def GetTFSession():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def CreateGameEnv():
    game = DoomGame()
    game.load_config("deadly_corridor.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()
    game.new_episode()
    ACTIONS = np.identity(7, dtype=int).tolist()
    return game, ACTIONS


def PreprocessFrame(frame):
    """
    Take a frame, crop the roof because it contains no useful information.
    Resize, normalize, and return preprocessedFrame. We don't grayscale the
    frame because it's already done in the vizdoom config.
    """
    crop = frame[15:-5, 20:-20]
    normalize = crop / 255.0
    resize = transform.resize(normalize,  [100, 120])
    return resize  # 100x120x1 frame


def InitDeque(length):
    """
    Initialize deque with zero-images, one array for each image
    """
    return deque([np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int)
                  for i in range(length)], maxlen=length)


def StackFrames(frames, state, isNewEpisode):
    """
    Stack frames together to give network a sense of time. If this is a new
    episode, duplicate the frame over the StackedState. OTW add this new frame
    to the existing queue.

    @param {*} state: A single frame from the game.

    @return {*} StackedState, frames
    """
    frame = PreprocessFrame(state)
    if isNewEpisode:
        frames = InitDeque(NUM_FRAMES)  # Clear frames
        # Because we're in a new episode, copy the same frame 4x
        frames.append(frame)
        frames.append(frame)
        frames.append(frame)
        frames.append(frame)
        StackedState = np.stack(frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        frames.append(frame)
        StackedState = np.stack(frames, axis=2)
    return StackedState, frames


game, ACTIONS = CreateGameEnv()

"""
MODEL PARAMETERS
"""

FRAME_WIDTH = 100
FRAME_HEIGHT = 120
NUM_FRAMES = 4  # Stack 4 frames together
STATE_SIZE = [FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES]
ACTION_SIZE = game.get_available_buttons_size()
LEARNING_RATE = 0.0002  # alpha
GAMMA = 0.95  # Discounting rate
UPDATE_TARGET_Q_EVERY = 10000

"""
TRAINING HYPERPARAMETERS
"""

NUM_PRETRAIN_EPS = 100
NUM_TRAIN_EPS = 500
NUM_TEST_EPS = 100
MAX_STEPS = 500  # Max possible steps in an episode
BATCH_SIZE = 64

DecayStep = 0

tf.reset_default_graph()
sess = GetTFSession()

frames = InitDeque(NUM_FRAMES)
agent = DDDQNet(sess, STATE_SIZE, ACTION_SIZE, LEARNING_RATE, name="agent")
targetNetwork = DDDQNet(sess, STATE_SIZE, ACTION_SIZE,
                        LEARNING_RATE, name="TargetNetwork")

sess.run(tf.global_variables_initializer())

MEMORY_SIZE = 100000
memory = Memory(MEMORY_SIZE)

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
SummaryOp = tf.summary.merge_all()

"""
TRAINING / TESTING
"""

TRAINING = True  # Set to False to test trained agent on games
if TRAINING == True:
    """
    PRETRAINING
    """

    for i in range(NUM_PRETRAIN_EPS):
        if i == 0:
            state = game.get_state().screen_buffer
            state, frames = StackFrames(frames, state, True)

        action = random.choice(ACTIONS)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            nstate = np.zeros(state.shape)
            experience = state, action, reward, nstate, done
            memory.store(experience)
            game.new_episode()
            state = game.get_state().screen_buffer
            state, frames = StackFrames(frames, state, True)
        else:
            nstate = game.get_state().screen_buffer
            nstate, frames = StackFrames(frames, nstate, False)
            experience = state, action, reward, nstate, done
            memory.store(experience)
            state = nstate

        if i % 100 == 0:
            print("Pretraining ep: {}".format(i))

    DecayStep = 0
    updateTargetQEvery = 0
    game.init()
    sess.run(targetNetwork.updateTargetGraph())

    for episode in range(NUM_TRAIN_EPS):
        step = 0
        done = False
        EpisodeReward = 0
        game.new_episode()
        state = game.get_state().screen_buffer
        state, frames = StackFrames(frames, state, True)

        while not done and step < MAX_STEPS:
            """
            GATHERING EXPERIENCE
            """

            step += 1
            updateTargetQEvery += 1
            DecayStep += 1
            action, ExploreProb = agent.PredictAction(DecayStep, state, ACTIONS)

            reward = game.make_action(action)
            done = game.is_episode_finished()
            EpisodeReward += reward

            if done:
                nstate = np.zeros((FRAME_WIDTH, FRAME_HEIGHT), dtype=np.int)
                nstate, frames = StackFrames(frames, nstate, False)
                experience = state, action, reward, nstate, done
                memory.store(experience)
                print('Episode: {}'.format(episode),
                      'Reward: {}'.format(EpisodeReward),
                      'Loss: {:.4f}'.format(loss),
                      'Explore Prob: {:.4f}'.format(ExploreProb))
            else:
                nstate = game.get_state().screen_buffer
                nstate, frames = StackFrames(frames, nstate, False)
                experience = state, action, reward, nstate, done
                memory.store(experience)
                state = nstate

            """
            LEARNING
            """

            treeIdx, batch, ISWeightsMB = memory.sample(BATCH_SIZE)

            statesMB = np.array([each[0][0] for each in batch], ndmin=3)
            actionsMB = np.array([each[0][1] for each in batch])
            rewardsMB = np.array([each[0][2] for each in batch])
            nstatesMB = np.array([each[0][3] for each in batch], ndmin=3)
            doneMB = np.array([each[0][4] for each in batch])

            """
            Double DQN. Instead of calculating the target Q values like we did
            before, we get the target from the TargetNetwork, which is a
            periodic clone of the agent and thus fixes its target longer for
            more stable training.
            """

            targetQsBatch = []

            Qs = sess.run(agent.output, feed_dict={agent.inputs: nstatesMB})
            QTargetNstate = sess.run(targetNetwork.output, feed_dict={
                targetNetwork.inputs: nstatesMB})

            for i in range(0, len(batch)):
                isdone = doneMB[i]
                action = np.argmax(Qs[i])
                if isdone:
                    targetQsBatch.append(rewardsMB[i])
                else:
                    target = rewardsMB[i] + GAMMA * QTargetNstate[i][action]
                    targetQsBatch.append(target)

            targetsMB = np.array([each for each in targetQsBatch])

            _, loss, AbsoluteErrors = sess.run([
                agent.optimizer, agent.loss, agent.AbsoluteErrors],
                feed_dict={agent.inputs: statesMB,
                           agent.targetQ: targetsMB,
                           agent.actions: actionsMB,
                           agent.ISWeights: ISWeightsMB})

            memory.batch_update(treeIdx, AbsoluteErrors)

            summary = sess.run(SummaryOp, feed_dict={
                agent.inputs: statesMB,
                agent.targetQ: targetsMB,
                agent.actions: actionsMB,
                agent.ISWeights: ISWeightsMB})
            writer.add_summary(summary, episode)
            writer.flush()

            if updateTargetQEvery > UPDATE_TARGET_Q_EVERY:
                sess.run(targetNetwork.updateTargetGraph())
                updateTargetQEvery = 0
                print("Model updated")

        if episode % 10 == 0:
            agent.save(SAVE_DIR + "/agent", episode)
else:
    game = DoomGame()
    game.load_config("deadly_corridor_testing.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()

    for i in range(NUM_TEST_EPS):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, frames = StackFrames(frames, state, True)

        while not game.is_episode_finished():
            ExploreProb = 0.01
            if (ExploreProb > np.random.rand()):
                action = random.choice(ACTIONS)
            else:
                Qs = sess.run(agent.output, feed_dict={
                    agent.inputs: state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
                action = ACTIONS[int(choice)]

            game.make_action(action)
            done = game.is_episode_finished()

            if done:
                break
            else:
                nstate = game.get_state().screen_buffer
                nstate, frames = StackFrames(frames, nstate, False)
                state = nstate

        score = game.get_total_reward()
        print("Score: ", score)

    game.close()
