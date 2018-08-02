# coding=utf-8
import paddle
import paddle.fluid as fluid
import paddle.dataset.mnist as mnist
import numpy


def softmax_regression():
    img = fluid.layers.data(name='img', shape=[1, 28, 28])
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict


def multilayer_perceptron():
    img = fluid.layers.data(name='img', shape=[1, 28, 28])
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def convolutional_neural_network():
    img = fluid.layers.data(name='img', shape=[1, 28, 28])
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img, filter_size=5, num_filters=20,
        pool_size=2, pool_stride=2, act='relu'
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1, filter_size=5, num_filters=20,
        pool_size=2, pool_stride=2, act='relu'
    )

    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = convolutional_neural_network()
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]


def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)


# 数据集设置
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500
    ),
    batch_size=64
)
test_reader = paddle.batch(
    paddle.dataset.mnist.test(), batch_size=64
)


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

trainer = fluid.Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program
)


params_dirname = 'recognize_digits_network.inference.model'
lists = []
def event_handler(event):
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 100 == 0:
            print('Pass %d, epoch %d, Cost %f' % (event.step, event.epoch, event.metrics[0]))

    if isinstance(event, fluid.EndEpochEvent):
        avg_cost, acc = trainer.test(reader=test_reader, feed_order=['img', 'label'])
        print('Test with Epoch %d, avg_cost: %s, acc: %s' % (event.epoch, avg_cost, acc))
        trainer.save_params(params_dirname)
        lists.append((event.epoch, avg_cost, acc))


from paddle.v2.plot import Ploter

train_title = 'Train cost'
test_title = 'Test cost'
cost_ploter = Ploter(train_title, test_title)
step = 0


def event_handler_plot(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        if step%100 == 0:
            cost_ploter.append(train_title, step, event.metrics[0])
            cost_ploter.plot()
            step += 1

    if isinstance(event, fluid.EndEpochEvent):
        trainer.save_params(params_dirname)
        avg_cost, acc = trainer.test(reader=test_reader, feed_order=['img', 'label'])
        cost_ploter.append(test_title, step, avg_cost)
        lists.append((event.epoch, avg_cost, acc))


trainer.train(num_epochs=5,
              event_handler=event_handler,
              reader=train_reader,
              feed_order=['img', 'label'])


