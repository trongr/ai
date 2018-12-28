import tensorflow as tf
import os
import model
import architecture as policies
import sonic_env as env
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def main():
    """
    Play/test script
    """
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Avoid warning messages
    config.gpu_options.allow_growth = True

    with tf.Session(config=config):
        model.play(policy=policies.A2CPolicy, env=DummyVecEnv([env.make_train_3]))


if __name__ == "__main__":
    main()

