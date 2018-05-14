#本文档读取手写数字图片，进行灰阶处理，然后读取文档6里学习完保存的W，b参数，运用softmax算法进行识别。可评判算法准确率。

import pickle
import tensorflow as tf
import numpy as np
from skimage import io

img = io.imread('hLabel9.png',as_grey=True)
img = np.array(img.reshape([-1,784]),dtype='float32')

pkl_file=open('parameter6.pkl','rb')
W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2 = pickle.load(pkl_file)

x = tf.constant(img)
W1 = tf.constant(W_conv1)
b1 = tf.constant(b_conv1)
W2 = tf.constant(W_conv2)
b2 = tf.constant(b_conv2)
W3 = tf.constant(W_fc1)
b3 = tf.constant(b_fc1)
W4 = tf.constant(W_fc2)
b4 = tf.constant(b_fc2)

#定义卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=(1,1,1,1),padding='SAME')

#定义池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=(1,2,2,1),padding='SAME')

x_image=tf.reshape(x,[-1,28,28,1])

#定义第一个卷积层
h_conv1 = tf.nn.relu(conv2d(x_image,W1)+b1)
h_pool1 = max_pool_2x2(h_conv1)

#定义第二个卷积层
h_conv2 = tf.nn.relu(conv2d(h_pool1,W2)+b2)
h_pool2 = max_pool_2x2(h_conv2)

#定义第三层为全连接层
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W3)+b3)

#将第三层的输出连接一个Softmax层，得到最后的概率输出
y = tf.nn.softmax(tf.matmul(h_fc1,W4)+b4)

z = tf.argmax(y,1)
sess=tf.Session()
print(sess.run(y))
print(sess.run(z))