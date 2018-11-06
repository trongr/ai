from time import sleep
import gym

LEFT = 0
RIGHT = 1


def SimplePolicyPosition(obs):
    """
    obs: Observation

    Go left when the cart is to the right of origin.
    Go right when the cart is to the left of origin.
    """
    pos = obs[0]
    if pos < 0:
        return RIGHT
    else:
        return LEFT


def SimplePolicyAngle(obs):
    """
    obs: Observation

    Go left when the pole is leaning to the left.
    Go right when the pole is leaning to the right.
    """
    angle = obs[2]
    if angle < 0:
        return LEFT
    else:
        return RIGHT


env = gym.make("CartPole-v0")

for i in range(100):
    obs = env.reset()

    for step in range(1000):
        action = SimplePolicyAngle(obs)
        obs, reward, done, info = env.step(action)

        env.render()

        if done == True:
            break

    print("Ep: {}. Run time: {}".format(i, step))
