刚开始训练loss就是nan,原因很可能是因为使用了cross entropy时出现了log(0),需要加一个epsilon防止该情况出现:
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) # 会有log(0)的问题
改为:
logits = tf.nn.log_softmax(logits)
loss = -tf.reduce_mean(labels*logits)

batch_size设置,可以先从128开始,然后根据结果以2倍往左右调

构造graph/network时,如卷积层的创建时都需要给定参数的初始化方法(在tf.get_variable里),
各种初始化方法: https://blog.csdn.net/FrankieHello/article/details/79781422
w一般用xavier方法,b一般是常数初始化为0,slim.conv2d这种函数里都会包含这些初始化设置; 
global_step要使用常数初始化为0;
当实施train时,需要先sess.run(tf.global_variables_initializer()),来实现初始化的op.

loss不变了: 可以增大一下学习率,

loss需要减小到多少: 简单二分类需要0.2以下

train的loss减小,accuracy增大,但是val相反,是过拟合了,改进方法:
1.增大数据数量
2.缩减网络:正则化,dropout. 一般dropout放在fc层前面,因为fc层的参数特别多,而卷积层的参数并不是那么多,如果不起作用,可以尝试在卷积层后也加dropout.

如何调用ckpt继续上一次的训练
saver=tf.train.Saver()
假设保存变量的时候是
checkpoint_filepath='models/train.ckpt'
saver.save(session,checkpoint_filepath)
则从文件读变量取值继续训练是
saver.restore(session,checkpoint_filepath)

在inference的时候,可能会遇到Variable already exists, disallowed. Did you mean to set reuse=True in VarScope?的问题
是因为:再次执行的时候，之前的计算图已经存在了，再次执行时会和之前已经存在的产生冲突。解决方法： 在代码前面加一句：tf.reset_default_graph（）
tf.reset_default_graph()不能在以下结构里使用:
Inside a with graph.as_default(): block.
Inside a with tf.Session(): block.
Between creating a tf.InteractiveSession and calling sess.close().


name / variable_scope???:https://blog.csdn.net/Jerr__y/article/details/70809528

深度学习模型的部署需要硬件上有配套的linux系统+python+tensorflow以及相关lib,会占用很大空间,对硬件要求很高,是不是需要考虑在线的解决方案?
