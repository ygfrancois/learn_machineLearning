from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import numpy as np
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # one_hot 是独热编码，使分类器更好处理属性数据，扩充特征


# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
#
# print(mnist.validation.images.shape)
# print(mnist.validation.labels.shape)
#
# print(mnist.test.images.shape)
# print(mnist.test.labels.shape)
#
# print(mnist.train.images[0, :])
# print(mnist.train.labels[0, :])

save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)


for i in range(20):
    image_array = mnist.train.images[i, :]
    image_array = image_array.reshape(28, 28)
    filename = save_dir + 'mnist_train_%d.jpg' % i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

    one_hot_label = mnist.train.labels[i, :]
    lable = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg lable: %d' % (i, lable))


x = tf.placeholder(tf.float32, [None, 784])
# 第三种placeholder相当于占位符，也是有shape的量，
# 因为训练过程中需要不断的赋值和替换值，而整体的计算结构是不变的，比如式1.1.1中X和C是不断的替换的，而A是不变的，这时就需要用到placeholder。
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax activation function

y_ = tf.placeholder(tf.float32, [None, 10])  # 实际图像标签

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print('start training...')

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
