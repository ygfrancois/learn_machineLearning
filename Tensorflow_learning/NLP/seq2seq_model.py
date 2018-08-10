import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper


class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, num_units,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forword_only=False):
        """
        构建模型
        :param source_vocab_size: 问句词汇表大小
        :param target_vocab_size: 答句词汇表大小
        :param buckets: (I,O),I指定最大输入长度,O指定最大输出长度
        :param num_units: 每一层神经元的数量
        :param num_layers:模型层数
        :param max_gradient_norm: 梯度被削减的最大
        :param batch_size:
        :param learning_rate:
        :param learning_rate_decay_factor:
        :param use_lstm:是否用lstm代替gru
        :param num_samples:使用softmax的样本数
        :param forword_only: 是否仅构建前向传播
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor
        )
        self.global_step = tf.Variable(0, trainable=False)

        embedding_encoder = tf.get_variable("embedding_encoder", [source_vocab_size, emb])

        # Encoder
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, enco
        )







        # output_projection = None
        # softmax_loss_function = None
        #
        # if 0 < num_samples < self.target_vocab_size:
        #     w = tf.get_variable("proj_w", [num_units, self.target_vocab_size])
        #     w_t = tf.transpose(w)
        #     b = tf.get_variable("proj_b", [self.target_vocab_size])
        #     output_projection = (w, b)
        #
        #     def sampled_loss(inputs, labels):
        #         labels = tf.reshape(labels, [-1, 1])
        #         return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
        #                                           self.target_vocab_size)
        #     softmax_loss_function = sampled_loss
        #
        # single_cell = tf.nn.rnn_cell.GRUCell(num_units)
        # if use_lstm:
        #     single_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        # cell = single_cell
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        # if num_layers > 1:
        #     cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        #
        # # Attention模型
        # def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        #     return attention_wrapper()
        #
        #
        # self.encoder_input = []
        # self.decoder_input = []
        # self.target_weights = []
        # # for i in range(buckets[-1][0]):


