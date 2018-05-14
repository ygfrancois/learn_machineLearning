# 本文档记录了第一个GYM CartPole学习算法，通过随机移动进行游戏，能坚持10多步到50步之间。

import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

# 游戏重启
env.reset()

random_episodes = 0
reward_sum = 0

while random_episodes < 50:

    # 画出图像
    env.render()

    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
env.reset()