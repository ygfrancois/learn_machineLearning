# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating,
as the compilation time can be a blocker using Theano.
Timings:
Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min
Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function

from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)
num_classes = 10


def build_generator(latent_size):
    # 构造一个generator，
    # 映射一组图像空间(格式为(..., 28, 28, 1))的参数（z, L）,
    # 其中z是分布的向量,L是从P_c里取的标记
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()  # 创建一个layer的线性栈,用于后面添加layers,相当于初始化一个卷积网络

    # dense构建网络层,当为第一层时,需要给出输入input_dim,不为第一层时不需要
    cnn.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
    cnn.add(Reshape((3, 3, 384)))

    # upsample to (7, 7, ...)
    cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    # upsample to (14, 14, ...)
    cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    # upsample to (28, 28, ...)
    cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal'))

    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(
    latent_size,))  # 创建先验分布 z space,一维长度为                    latent_size的噪声向量(特征向量),但并没有初始化(只是构建图) ,实际上是随机sample的一个特征向量
    # 这是一个condition GAN,所以除了噪声向量,还要构建标记:
    image_class = Input(shape=(1,), dtype='int32')  # 创建图像种类标记,一维整型向量,且长度也为1

    cls = Flatten()(Embedding(num_classes, latent_size, embeddings_initializer='glorot_normal')(image_class))
    # 先了解一下嵌入层的知识:http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/
    # 先创建一个嵌入层(Embedding()()的第一个括号里是argument,第二个括号里是输入),输入数据的长度为1,num_classes=10表示最大的输入是9,即输入是0-9的10个整数,即之前创建的图像标记image_class(不是独热数据类型,是单个的整数). latent_size代表全连接嵌入的维度,embedding层的输出shape为:(input_size, output_dim),这里的input_size由第二个括号里的image_class控制,即=1;output_dim=latent_size. 在这里latent_size就是输出的一维数据的长度(由于是全连接层,所以输入是1维,输出也是一维).
    # embedding初始化方法是glorot_normal法(xavier_normal法),个人理解是对image_class做初始化(之前只定义了是一维,长度为10,但并没有定义输入矩阵每个数的值),normal指正太分布,glorot算法可以通过输入和输出单元的数量自动确定权值矩阵的初始化大小,可以尽可能保证输入和输出数据有类似的概率分布. 其他初始化方法种类相关定义见:(http://keras-cn.readthedocs.io/en/latest/other/initializations/)
    # Flatten()()第一个括号是argument,第二个括号是输入,用于压平输入,将多维数据转成一维,这里由于Embedding的输出本身就是一维,所以实际上没有做操作.
    # 所以整个这行代码的作用是: 反向全连接层(与前面的反向卷积类似),得到正常卷积网络输出层前面的那层全连接层的输出, shape为(latent_size,)

    h = layers.multiply(
        [latent, cls])  # shape同为(latent_size,)的特征向量与标记嵌入向量乘,得到的还是shape为(latent_size,)的特征向量,该特征向量中加入了标记的信息,即condition.
    fake_image = cnn(h)  # h指generator的真实输入,通过构建的反向网络来生成image(生成出的假的)

    return Model([latent, image_class], fake_image)  # 创建函数式模型,input为[latent, image_class], output为fake_image


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    # 创建discriminator,这是一个正向卷积网络,输出为输入图像(generator生成出的或者真实的图像)的真假以及输入图像的分类(辅助分类器,实际上在GAN中只需要知道真假即可).
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))  # 使用了dropout,也可以不使用池化层

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(features)  # 0<fake<1,用于判定真假
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)  # aux是一个长度为10的概率分布,和为1,用于完成图像的分类

    return Model(image, [fake, aux])  # model的输入是image,输出是[fake, aux]


if __name__ == '__main__':

    # batch and latent size taken from the paper
    epochs = 100
    batch_size = 100
    latent_size = 100  # 特征向量的长度为100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    # 定义优化器的学习率和超参数
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()  # 构建判别网络
    # keras的模型通过Model.compile()来定义优化器,损失函数,这里对于真假二分类问题使用binary_crossentropy损失函数,对于图像代表的数字的10分类问题,采用sparse_categorical_crossentropy分类器.
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()  # 打印这个网络的日志

    # build the generator
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size,))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])  # 这些都是在构建图,并没有赋值计算

    # we only want to be able to train generation for the combined model
    # 先固定discriminator,来更新generator,让generator能够生成可以欺骗该discriminator的图片
    discriminator.trainable = False
    fake, aux = discriminator(fake)  # 计算出该fake的值(接近0或者1),和该fake的推测值
    combined = Model([latent, image_class],
                     [fake, aux])  # 输入为创建的[latent, image_class](图片的特征向量和标记),和generator生成的结果[fake, aux]

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1) with
    # range [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # 将像素值归一化,取值范围[-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # 将mnist数据转化为(...,28,28,1)维度

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    num_train, num_test = x_train.shape[0], x_test.shape[0]  # 训练和测试的数据个数

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
        progress_bar = Progbar(target=num_batches)

        # we don't want the discriminator to also maximize the classification
        # accuracy of the auxiliary classifier on generated images, so we
        # don't train discriminator to produce class labels for generated
        # images (see https://openreview.net/forum?id=rJXTf9Bxg).
        # To preserve sum of sample weights for the auxiliary classifier,
        # we assign sample weight of 2 to the real images.
        disc_sample_weight = [np.ones(2 * batch_size),
                              np.concatenate((np.ones(batch_size) * 2,
                                              np.zeros(batch_size)))]

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size,
                                              latent_size))  # 随机给噪声向量取值,范围是[-1,1),这与之前对mnist数据集做的处理的范围一样;shape是(batch_size, latent_size),这与全连接层之前的特征向量一样

            # get a batch of real images
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, num_classes, batch_size)  # 给噪声随机取一些噪声标记,范围是[0,10)里的整数,个数是batch_size

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            # 通过创建的CGAN生成虚假images
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch, generated_images))  # 将真实图片和虚假图片并列放到x里,real图片占前半部分,fake占后半部分

            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array(
                [soft_one] * batch_size + [soft_zero] * batch_size)  # 判断真假的标记值(前半部分真实的都取0.95(逼近1),后半部分假的都取0(不给假的任何机会))
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)  # 相应的分类的标记

            # see if the discriminator can figure itself out...
            # sample_weight定义每个sample loss的权重,这里discriminator只需要判断真假,不需要鉴别种类,鉴别种类只是个辅助.
            # 所以: y是2个batch的真假标记,loss的权重都设置为1;aux_y是两个batch的分类标记,对real的数据设置loss权重为2,对fake的数据设置loss权重为0,
            # 因为对于假图片,我们没有必要去关心他们的辅助分类结果来更新discriminator,改善discriminator的辅助分类效果只需要着重于real图片.
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size) * soft_one

            # 输入里有sampled_labels,用作condition,输出里,sampled出来的标记都被当做正确的,所以标记分别就是trick和sampled_labels本身
            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

            progress_bar.update(index + 1)  # batch数进度条跟新一下

        print('Testing for epoch {}:'.format(epoch))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (num_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        x = np.concatenate((x_test, generated_images))
        y = np.array([1] * num_test + [0] * num_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        num_rows = 40
        noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)),
                        (num_classes, 1))

        sampled_labels = np.array([
            [i] * num_rows for i in range(num_classes)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # prepare real images sorted by class label
        real_labels = y_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes]
        indices = np.argsort(real_labels, axis=0)
        real_images = x_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes][indices]

        # display generated images, white separator, real images
        img = np.concatenate(
            (generated_images,
             np.repeat(np.ones_like(x_train[:1]), num_rows, axis=0),
             real_images))

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(img, 2 * num_classes + 1)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))

    with open('acgan-history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)
