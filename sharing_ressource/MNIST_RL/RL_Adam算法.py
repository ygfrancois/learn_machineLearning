# 本文档记录了第二个GYM CartPole学习算法，通过Adam算法，进行游戏。


import numpy as np
import tensorflow as tf
import gym



env = gym.make('CartPole-v0')

H = 50
batch_size = 25
learning_rate = 0.1
D = 4
gamma = 0.99

observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# 优化器使用Adam算法
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


# 定义潜在奖励价值
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# 定义虚拟标签
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)

tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True
        #print(reward_sum / batch_size)
        x =np.reshape(observation, [1, D])



        tfprob = sess.run(probability, feed_dict={observations:x})
        action = 1 if np.random.uniform() < tfprob else 0

        xs.append(x)
        y = 1 - action
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []

            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            tGrad = sess.run(newGrads,feed_dict={observations:epx, input_y:epy,advantages:discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size ==0:
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix]=grad * 0

                print('Average reward for epsiode %d : %f.'%(episode_number,reward_sum/batch_size))

                if reward_sum/batch_size>200:
                    print("Task solve in",epsiode_numbe,'episodes1')
                    break
                reward_sum  = 0
            observation = env.reset()
