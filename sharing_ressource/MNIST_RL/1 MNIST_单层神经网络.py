# 本文档记录了第一个Tensorflow学习算法，通过softmax算法学习手写数字识别，准确率达到90%


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle


mnist = input_data.read_data_sets("D:/DISK e/VB and C/Python/MNist/Material/",one_hot=True)


# 1 定义算法公式，也就是神经网络Forward时的计算
# y=softmax(Wx+b)
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)


# 2 定义Loss，选定优化器，并指定优化器优化Loss
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

# 优化器
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)


# 3 迭代地对数据进行训练
for i in range(1000):

    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

# 4 在测试集或验证集上对准确率进行评测
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

# 将学习完的参数W b 保存到文件
output =open('parameter.pkl','wb')
pickle.dump([sess.run(W),sess.run(b)],output)
output.close()
