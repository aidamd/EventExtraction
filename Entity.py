import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import f1_score
import math


class Entity():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embedding_placeholder = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                                     trainable=False, name="W")
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.embedding_placeholder = self.build_embedding()
        self.sequence_length = tf.placeholder(tf.int64, [None])
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.batch_size = tf.placeholder(tf.int64)

        self.target_group = tf.placeholder(tf.int64, [None])
        self.target_weight = tf.placeholder(tf.float64)
        self.hate_act = tf.placeholder(tf.int64, [None])
        self.act_weight = tf.placeholder(tf.float64)

        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

        self.keep_prob = tf.placeholder(tf.float32)

        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        self.network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.network, self.network, self.embed,
                                                              dtype=tf.float32, sequence_length=self.sequence_length)
        fw_outputs, bw_outputs = bi_outputs
        fw_states, bw_states = bi_states

        self.state = tf.concat([fw_states, bw_states], 2)
        self.state = tf.expand_dims(tf.reshape(self.state, [-1, 2 * self.hidden_size, 1]), 0)
        pooled_outputs = list()

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, 2 * self.hidden_size, 1, self.entity_num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[self.entity_num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(self.state, W, strides=[1, 1, 1, 1], padding="SAME")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))

            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            pooled_outputs.append(pooled)

        num_filters_total = self.entity_num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        fc_target = fully_connected(self.h_pool_flat, 8)
        fc_act = fully_connected(self.h_pool_flat, 5)

        self.high_target = tf.expand_dims(tf.reduce_max(fc_target, axis=0), 0)
        self.high_act = tf.expand_dims(tf.reduce_max(fc_act, axis=0), 0)

        t_weight = tf.gather(self.target_weight, self.target_group)
        a_weight = tf.gather(self.act_weight, self.hate_act)

        self.target_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=self.target_group,
                                                                      logits=self.high_target,
                                                                      weights=t_weight)
        self.act_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=self.hate_act,
                                                                   logits=self.high_act,
                                                                   weights=a_weight)

        self.loss = tf.add(self.target_xentropy, self.act_xentropy)

        self.predicted_target = tf.argmax(self.high_target, 1)
        self.predicted_act = tf.argmax(self.high_act, 1)

        self.accuracy_target = tf.reduce_mean(
              tf.cast(tf.equal(self.predicted_target, self.target_group), tf.float32))
        self.accuracy_act = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_act, self.hate_act), tf.float32))

        self.accuracy = (self.accuracy_target + self.accuracy_act) / 2
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.entity_learning_rate).minimize(self.loss)

    def run_model(self, batches, dev_batches, test_batches, weights):
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            #init.run()
            self.sess.run(init)
            epoch = 1
            target_weight, act_weight = weights
            while True:
                ## Train
                epoch_loss = float(0)
                epoch += 1
                train_accuracy = 0
                for batch in batches:
                    X_batch, X_len, _, target, act = batch
                    if len(X_batch) == 1:
                        continue
                    feed_dict = {self.train_inputs: X_batch,
                                self.sequence_length: X_len,
                                self.keep_prob: self.entity_keep_ratio,
                                self.target_group: [target],
                                self.hate_act: [act],
                                self.target_weight: target_weight,
                                self.act_weight: act_weight,
                                self.batch_size: len(X_batch)
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    loss_val, _, xen, xen2 = self.sess.run([self.loss, self.training_op, self.act_xentropy, self.predicted_act], feed_dict= feed_dict)
                    #print(xen)
                    train_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    if math.isnan(loss_val):
                        print()
                    epoch_loss += loss_val
                ## Test
                test_accuracy = 0
                t_pred, a_pred, t_true, a_true = list(), list(), list(), list()
                for X_batch, X_len, _, target, act in dev_batches:
                    feed_dict = {self.train_inputs: X_batch,
                                 self.sequence_length: X_len,
                                 self.keep_prob: 1,
                                 self.target_group: [target],
                                 self.hate_act: [act],
                                 self.batch_size: len(X_batch)
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    test_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    try:
                        target_, act_  = self.sess.run([self.predicted_target, self.predicted_act], feed_dict=feed_dict)
                        t_pred.extend(list(target_))
                        a_pred.extend(list(act_))
                        t_true.append(target)
                        a_true.append(act)
                    except Exception:
                        print()
                print(epoch, "Train accuracy:", train_accuracy / len(batches),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", test_accuracy / len(dev_batches),
                      "Act F1: ", f1_score(a_true, a_pred, average="macro"),
                      "Target F1: ", f1_score(t_true, t_pred, average="macro"))
                if epoch == self.epochs:
                    break
        return