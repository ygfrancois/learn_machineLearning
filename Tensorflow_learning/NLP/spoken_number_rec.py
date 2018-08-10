import tflearn
import tensorflow as tf
from tensorflow_speech_recognition_demo import speech_data
import numpy

learning_rate = 0.0001
training_iters = 3000
batch_size = 64

width = 20  # MFCC(梅尔频率倒谱系数Mel frequency cepstral coefficients)特征
height = 80  # 最大发音长度
classes = 10  # 数字类别

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y


net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
while training_iters > 0:  # training_iters
    model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=batch_size)
    _y = model.predict(X)

model.save("tflearn.lstm.model")


if __name__=="__main__":
    test_file = "/home/yangguang/machineLearning/learn_machineLearning/Tensorflow_learning/NLP/data/spoken_numbers_pcm/0_Albert_100.wav"
    test = speech_data.load_wav_file(test_file)
    model = model.load("tflearn.lstm.model")
    res = model.predict([test])
    res = numpy.argmax(res)
    print(res)