#本文档读取手写数字图片，进行灰阶处理，然后读取文档1里学习完保存的W，b参数，运用softmax算法进行识别。可评判算法准确率。

import pickle
import tensorflow as tf
import numpy as np
from skimage import io

#读取手写数字图片
img = io.imread('7.bmp',as_grey=True)
img = np.array(img.reshape([-1,784]),dtype='float32')

#读取文档1里学习完保存的W，b参数
pkl_file=open('parameter.pkl','rb')
w,b=pickle.load(pkl_file)

x = tf.constant(img)
W = tf.constant(w)
B = tf.constant(b)
y = tf.nn.softmax(tf.matmul(x,W)+B)
z = tf.argmax(y,1)
sess=tf.Session()
print(sess.run(y))
print(sess.run(z))