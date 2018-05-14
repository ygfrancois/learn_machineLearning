#本文档读取手写数字图片，进行灰阶处理，然后读取文档4里学习完保存的W，b参数，运用softmax算法进行识别。可评判算法准确率。

import pickle
import tensorflow as tf
import numpy as np
from skimage import io

img = io.imread('9.png',as_grey=True)
img = np.array(img.reshape([-1,784]),dtype='float32')

pkl_file=open('parameter.pkl','rb')
w1,b1,w2,b2=pickle.load(pkl_file)

x = tf.constant(img)
W1 = tf.constant(w1)
B1 = tf.constant(b1)
W2 = tf.constant(w2)
B2 = tf.constant(b2)


# 隐含层公式 y=relu(W1x+b1)
hidden1=tf.nn.relu(tf.matmul(x,W1)+B1)

# y = tf.nn.softmax(tf.matmul(hidden1,W)+B)
y=tf.nn.softmax(tf.matmul(hidden1,W2)+B2)

z = tf.argmax(y,1)
sess=tf.Session()
print(sess.run(y))
print(sess.run(z))