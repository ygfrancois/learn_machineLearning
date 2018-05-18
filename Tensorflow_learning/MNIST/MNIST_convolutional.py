import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


if __name__=='__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784], name='x')  # 需要添加参数name=“x”来定义operation的name
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])  # shape数组里的第一个参数表示一个batch的图片数量，由于此处数量未知，
                                              # 所以用-1自动补齐，第4个参数表示in_channel数，此处黑白为1

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # pooling 之后图片变成14*14*32

    # second conv layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  # pooling 之后图片变成7*7*64,相当于这么多特征值

    # full connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fcl 2
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    b_constant = tf.constant(value=1, dtype=tf.float32)  # 用于下行给logits赋名字
    logits = tf.multiply(y_conv, b_constant, name='logits')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # 此处的logits是没有softmax过的数据
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 此处获得一个True和False构成的数组
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 存储训练过程中的模型
    ckpt_dir = "./MNIST_conv_model"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()
    non_storable_variable = tf.Variable(777)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = mnist.train.next_batch(50)

        if i%100 ==0:  # 每进行100步输出当前步已经训练出的网络的准确率
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})  # eval起所有步骤
            print("step %d, training accuracy %g" % (i, train_accuracy))

            global_step.assign(i).eval()
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

