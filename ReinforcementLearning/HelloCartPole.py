import time
import gym

env = gym.make('CartPole-v0')
print("ACTION SPACE", env.action_space)
print("OBSERVATION SPACE", env.observation_space)

for i_episode in range(20):
    observation = env.reset()
    for t in range(3):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
