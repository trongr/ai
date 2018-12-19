import tensorflow as tf
import numpy as np
from vizdoom import *
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt


def create_environment():
    game = DoomGame()
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    game.init()
    # Here our possible actions # [[1,0,0],[0,1,0],[0,0,1]]
    possible_actions = np.identity(3, dtype=int).tolist()
    return game, possible_actions


def preprocess_frame(frame):
    cropped_frame = frame[80:, :]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int)
                                for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
            self.actions = tf.placeholder(
                tf.int32, [None, action_size], name="actions")
            self.discounted_episode_rewards_ = tf.placeholder(
                tf.float32, [None, ], name="discounted_episode_rewards_")

            # Add this placeholder for having this variable in tensorboard
            self.mean_reward_ = tf.placeholder(
                tf.float32, name="mean_reward")

            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs_,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training=True,
                epsilon=1e-5,
                name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training=True,
                epsilon=1e-5,
                name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")

            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training=True,
                epsilon=1e-5,
                name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")

            self.logits = tf.layers.dense(
                inputs=self.fc,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                units=3,
                activation=None)

            self.action_distribution = tf.nn.softmax(self.logits)

            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.actions)
            self.loss = tf.reduce_mean(
                self.neg_log_prob * self.discounted_episode_rewards_)

            self.train_opt = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)


def make_batch(batch_size, stacked_frames):
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    episode_num = 1
    game.new_episode()
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while True:
        action_probability_distribution = sess.run(
            PGNetwork.action_distribution, feed_dict={
                PGNetwork.inputs_: state.reshape(1, *state_size)})

        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())
        action = possible_actions[action]

        reward = game.make_action(action)
        done = game.is_episode_finished()

        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)

        if done:
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(
                stacked_frames, next_state, False)

            rewards_of_batch.append(rewards_of_episode)

            discounted_rewards.append(
                discount_and_normalize_rewards(rewards_of_episode))

            # If the number of rewards_of_batch > batch_size stop the minibatch
            # creation (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire
            # episode (Monte Carlo) so we can't check that condition for each
            # step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break

            rewards_of_episode = []
            episode_num += 1
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(
                stacked_frames, next_state, False)
            state = next_state

    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num


game, possible_actions = create_environment()
stack_size = 4
stacked_frames = deque([np.zeros((84, 84), dtype=np.int)
                        for i in range(stack_size)], maxlen=4)
state_size = [84, 84, stack_size]
action_size = game.get_available_buttons_size()
stack_size = 4
learning_rate = 0.002
num_epochs = 500
batch_size = 5000
gamma = 0.95  # Discounting rate
training = True

tf.reset_default_graph()
PGNetwork = PGNetwork(state_size, action_size, learning_rate)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter("./logs/")
tf.summary.scalar("Loss", PGNetwork.loss)
tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_)
write_op = tf.summary.merge_all()

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []
saver = tf.train.Saver()
savepath = "./save/model.ckpt"
# saver.restore(sess, savepath)

if training:
    while epoch < num_epochs + 1:
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(
            batch_size, stacked_frames)

        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)

        mean_reward_of_that_batch = np.divide(
            total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)

        average_reward_of_all_training = np.divide(
            np.sum(mean_reward_total), epoch)

        maximumRewardRecorded = np.amax(allRewards)

        summary, loss_, _ = sess.run([
            write_op, PGNetwork.loss, PGNetwork.train_opt], feed_dict={
            PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
            PGNetwork.actions: actions_mb,
            PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
            PGNetwork.mean_reward_: mean_reward_of_that_batch
        })

        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(
            total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(
            average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))
        print("Training Loss: {}".format(loss_))

        writer.add_summary(summary, epoch)
        writer.flush()

        if epoch % 10 == 0:
            saver.save(sess, savepath)
            print("Model saved")
        epoch += 1
