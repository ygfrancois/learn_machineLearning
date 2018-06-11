import tensorflow as tf
from skimage import io, transform
import numpy as np
# 模型调用方法参考：https://www.2cto.com/kf/201707/654487.html
import scipy.misc


path1 = "/home/yangguang/machineLearning/learn_machineLearning/Tensorflow_learning/MNIST/t8.jpg"


def read_one_image(path):
    img = io.imread(path, as_grey=True)  # just read gray channel
    img = img.reshape(1, 784)
    return np.asarray(img)

data = read_one_image(path1)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

ckpt_dir = "./MNIST_conv_model/"
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
if ckpt and ckpt.model_checkpoint_path:
    print(ckpt.model_checkpoint_path)
    saver = tf.train.import_meta_graph(ckpt_dir + "model.ckpt-19900.meta")
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")  # x是节点名称，x:0是tensor名称
    y_conv = graph.get_tensor_by_name("logits:0")
    y = graph.get_tensor_by_name("Placeholder_1:0")

    # x = tf.placeholder(tf.float32, [1, 784])
    # y = tf.placeholder(tf.float32, [1, 10])
    y_data = np.asarray(1)

    result = sess.run(y_conv, feed_dict={x: data, y:y_data})
    print(result)

    print(tf.argmax(result, 1).eval())