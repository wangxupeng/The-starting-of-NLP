import gc
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from tensorflow.python.keras._impl.keras.initializers import he_uniform
from tensorflow.python.layers.convolutional import separable_conv1d
import pandas as pd
import sys

from tensorflow.python.ops.rnn_cell_impl import GRUCell

flags = tf.flags
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", True, "Log_device_placement")
flags.DEFINE_integer('batch_size', 128, 'model batch_size')
flags.DEFINE_integer('num_classes', 19, 'the number of label classes')
flags.DEFINE_integer('num_filters', 128, 'the number of filters')
flags.DEFINE_integer('num_words', 400000, 'the number of most important words in all article')
flags.DEFINE_integer('maxlen', 1500, 'the number of data length after tokenizer and pad')
flags.DEFINE_integer('n_epochs', 50, 'the number of epochs')
flags.DEFINE_integer('DIM', 256, 'the dim of embedding vector')
flags.DEFINE_string('mode', "model_parallel", 'here are two mode--model_parallel and data_parallel')
FLAGS = flags.FLAGS


def bulid_model(sentences, num_words, DIM, num_classes, labels, num_filters):
    with tf.device("cpu:0"):
        # The embedding layer run on cpu
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                     shape=[num_words, DIM],
                                     initializer=tf.random_normal_initializer(stddev=0.005, seed=1))
        sentences = tf.nn.embedding_lookup(embeddings, sentences, )

    with tf.variable_scope('bigru_layer', reuse=tf.AUTO_REUSE):
        lstm_fw_cell = tf.contrib.rnn.GRUCell(num_units=num_filters)  # forward direction cell
        lstm_bw_cell = tf.contrib.rnn.GRUCell(num_units=num_filters)  # backward direction cell
        cells_fw = [lstm_fw_cell for _ in range(1)]
        cells_bw = [lstm_bw_cell for _ in range(1)]
        bigru_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, sentences,
                                                                             dtype=tf.float32)

    with tf.variable_scope("attetion", reuse=tf.AUTO_REUSE):
        attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[num_filters * 2],
                                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # 全连接层，把 h_i 转为 u_i ， shape= [batch_size, units, input_size] -> [batch_size, units, output_size]
        input_projection = tf.contrib.layers.fully_connected(bigru_outputs, num_filters* 2, activation_fn=tf.tanh)
        # 输出 [batch_size, units]
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)

        weighted_projection = tf.multiply(bigru_outputs, attention_weights)
        attentnion_outputs = tf.reduce_sum(weighted_projection, axis=1)

        bn = tf.layers.batch_normalization(attentnion_outputs,
                                           momentum=0.90,
                                           training=True,
                                           )

        fully_connect = tf.layers.dense(bn,
                                        num_filters,
                                        activation=tf.nn.relu)

        drop_out = tf.layers.dropout(fully_connect, 0.2)

        W = tf.get_variable(
            "W",
            shape=[num_filters, num_classes],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_classes]))
        logits = tf.nn.xw_plus_b(drop_out, W, b, name="text_cnn_logits")

    with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                            labels=labels)
        loss = tf.reduce_mean(losses)

    with tf.variable_scope("opt", reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(0.000001)
        grads = optimizer.compute_gradients(loss)

    model_spec = {}
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['logits'] = logits
    model_spec['loss'] = loss
    model_spec["grad"] = grads
    return model_spec


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def num_batch_size(data):
    data_len = len(data)
    num_batch = int((data_len - 1) / FLAGS.batch_size) + 1

    return num_batch, data_len


def f1_avg(y_pred, y_true):
    '''
    mission 1&2
    :param y_pred: predict labels
    :param y_true: true value labels
    :return:
    '''
    f1_micro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='micro')
    f1_macro = f1_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='macro')
    return (f1_micro + f1_macro) / 2


def main(_):
    config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                            log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    DIM = FLAGS.DIM
    num_filters = FLAGS.num_filters
    num_words = FLAGS.num_words
    maxlen = FLAGS.maxlen
    n_epochs = FLAGS.n_epochs
    mode = FLAGS.mode
    print('num_words = {}, maxlen = {}, mode = {}'.format(num_words, maxlen, mode))

    # data and labels
    fact = np.load(r"./data/train_word_seg_data{}_numwords{}.npy".format(maxlen, num_words))
    labels = np.load(r"./data/train_label_from_zero_onehot.npy")
    fact_train, fact_test, labels_train, labels_test = train_test_split(fact, labels, test_size=0.1,
                                                                        random_state=1)
    del fact
    del labels
    gc.collect()

    tf.reset_default_graph()

    X = tf.placeholder(tf.int32, [None, maxlen], name="X")
    y = tf.placeholder(tf.int8, [None, num_classes], name="y")

    grads = []
    loss = []
    lastlayer = []

    if mode == "data_parallel":
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(2):
                with tf.device("/gpu:{}".format(i)):
                    counter = 0
                    sentences = X[i * batch_size: (i + 1) * batch_size]
                    labels = y[i * batch_size: (i + 1) * batch_size]
                    model = bulid_model(sentences, num_words, DIM, num_classes, labels, num_filters)
                    tf.get_variable_scope().reuse_variables()
                    counter += 1
                    grads.append(model["grad"])
                    loss.append(model['loss'])
                    lastlayer.append(model["logits"])

    if mode == "model_parallel":
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(2):
                with tf.device("/gpu:{}".format(i)):
                    sentences = X
                    labels = y
                    model = bulid_model(sentences, num_words, DIM, num_classes, labels, num_filters)
                    tf.get_variable_scope().reuse_variables()
                    grads.append(model["grad"])
                    loss.append(model['loss'])
                    lastlayer.append(model["logits"])

    grads = average_gradients(grads)

    loss = tf.add_n(loss, name="total_loss")

    initial_learning_rate = 0.001
    decay_steps = 300
    decay_rate = 1 / 10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    apply_gradient = optimizer.apply_gradients(grads, global_step=global_step)

    if mode == "model_parallel":
        lastlayer = tf.add_n(lastlayer)

    if mode == "data_parallel":
        lastlayer = tf.concat(lastlayer, axis=0)

    predict = tf.argmax(lastlayer, 1, name="text_cnn_predictions")
    ture_y = tf.argmax(y, 1)
    correct = tf.equal(predict, ture_y)
    accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

    init = tf.global_variables_initializer()

    num_batch, data_len = num_batch_size(fact_train)

    with tf.Session(config=config) as sess:
        init.run()
        for epoch in range(n_epochs):
            for i in range(num_batch):
                start_id = i * batch_size

                end_id = min((i + 1) * batch_size, data_len)

                X_batch, y_batch = fact_train[start_id:end_id], labels_train[start_id:end_id]

                sess.run(apply_gradient, feed_dict={X: X_batch, y: y_batch})

                if i % 100 == 0:
                    acc_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
                    train_y_hat, train_y = sess.run([predict, ture_y], feed_dict={X: X_batch, y: y_batch})
                    f1 = f1_avg(train_y_hat, train_y)
                    lr = sess.run([learning_rate], feed_dict={X: X_batch, y: y_batch})

                    num_batch_valid, data_len_valid = num_batch_size(fact_test)
                    avg_acc_valid = []
                    avg_f1_valid = []

                    for k in range(num_batch_valid):
                        start_id_valid = k * batch_size
                        end_id_valid = min((k + 1) * batch_size, data_len_valid)
                        X_batch_valid, y_batch_valid = fact_test[start_id_valid:end_id_valid], labels_test[
                                                                                               start_id_valid:end_id_valid]

                        acc_valid = sess.run(accuracy, feed_dict={X: X_batch_valid, y: y_batch_valid})
                        avg_acc_valid.append(acc_valid)

                        valid_y_hat, valid_y = sess.run([predict, ture_y],
                                                        feed_dict={X: X_batch_valid, y: y_batch_valid})
                        f1_valid = f1_avg(valid_y_hat, valid_y)
                        avg_f1_valid.append(f1_valid)

                    print("epoch", epoch, "Iteration", i,
                          "Train accuracy:{} Train F1:{}".format(acc_train, f1),
                          "Valid accuracy:{} Valid F1:{}".format(np.mean(avg_acc_valid), np.mean(avg_f1_valid)),
                          "learning_rate", lr,
                          )

                    del train_y
                    del train_y_hat
                    del avg_acc_valid
                    del avg_f1_valid
                    del acc_train
                    gc.collect()


if __name__ == '__main__':
    tf.app.run()
