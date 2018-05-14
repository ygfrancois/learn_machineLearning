
#本文档算法将MNist里的图片读出，整理成28*28的格式，存成图片，并命名格式里加上Label。读取图片时随机读取。

from tensorflow.examples.tutorials.mnist import input_data
from skimage import io,data

mnist = input_data.read_data_sets("D:/DISK e/VB and C/Python/MNist/Material/",one_hot=True)

for i in range(50):
    batch_xs,batch_ys=mnist.train.next_batch(1)
    x_temp=batch_xs.reshape((28,28))
    y_temp = batch_ys.tolist()

    f_name = '数字训练' + str(i) +'Label'+str(y_temp[0].index(max(y_temp[0])))+ '.png'
io.imsave(f_name,x_temp)