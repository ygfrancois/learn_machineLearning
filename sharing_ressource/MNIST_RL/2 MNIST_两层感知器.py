# 本文档记录了第二个Tensorflow学习算法，通过增加一层隐含层，再通过softmax算法学习手写数字识别，准确率达到98%

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle

mnist = input_data.read_data_sets("D:/DISK e/VB and C/Python/MNist/Material/",one_hot=True)
sess=tf.InteractiveSession()

# 1 定义算法公式，也就是神经网络Forward时的计算

in_units=784
h1_units=300
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)

# 隐含层公式 y=relu(W1x+b1)
hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)

y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)

# 2 定义Loss，选定优化器，并指定优化器优化Loss
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

# 优化器
train_step=tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)

init=tf.global_variables_initializer()
sess.run(init)

# 3 迭代地对数据进行训练
tf.global_variables_initializer().run()
for i in range(8000):
    batch_xs, batch_ys = mnist.train.next_batch(200)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.7})

# 4 在测试集或验证集上对准确率进行评测
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels,keep_prob:1.0}))
# 将学习完的参数W b 保存到文件
output =open('parameter.pkl','wb')
pickle.dump([sess.run(W1),sess.run(b1),sess.run(W2),sess.run(b2)],output)
output.close()