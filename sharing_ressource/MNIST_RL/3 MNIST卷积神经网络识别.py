# 本文档记录了第三个Tensorflow学习算法，通过增加两个卷积层加一个全连接层，实现卷积神经网络，再通过softmax算法学习手写数字识别，准确率达到99.2%

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle

mnist = input_data.read_data_sets("D:/DISK e/VB and C/Python/MNist/Material/",one_hot=True)
sess=tf.InteractiveSession()

# 1 定义算法公式，也就是神经网络Forward时的计算

# 定义全重W
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#定义偏置b
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=(1,1,1,1),padding='SAME')

#定义池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=(1,2,2,1),padding='SAME')

#定义输入的placeholder

x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

#定义第一个卷积层
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#定义第二个卷积层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#定义第三层为全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#为了减轻过拟合，使用一个Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#将Dropout层的输出连接一个Softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 2 定义Loss，选定优化器，并指定优化器优化Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
# 优化器
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
# 定义准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 3 迭代地对数据进行训练
tf.global_variables_initializer().run()
for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.5})
    #每100次训练，对准确率进行一次评测
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

# 4 在测试集或验证集上对准确率进行评测
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels,keep_prob:1.0}))

# 将学习完的参数W b 保存到文件
output =open('parameter6.pkl','wb')
pickle.dump([sess.run(W_conv1),sess.run(b_conv1),sess.run(W_conv2),sess.run(b_conv2),sess.run(W_fc1),sess.run(b_fc1),sess.run(W_fc2),sess.run(b_fc2)],output)
output.close()