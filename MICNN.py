import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import f1_score


class MICNN():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), self.embedding_size]),
                                     trainable=False, name="W")

            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocab), self.embedding_size])
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build(self):
        tf.reset_default_graph()
        self.sequence_length = tf.placeholder(tf.int64, [None])
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.embedding_placeholder = self.build_embedding()
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.output = tf.placeholder(tf.int64, [None])
        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)

        embed = tf.expand_dims(self.embed, axis=3)

        pooled_outputs = list()

        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]))
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

            conv = tf.nn.conv2d(embed, W, strides=[1, 1, 1, 1], padding="VALID")
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))

            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)

        # local vector for each sentence
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.batch_size, num_filters_total, 1])

        art_pooled_outputs = list()

        for i, art_filter_size in enumerate(self.art_filter_sizes):
            filter_shape = [art_filter_size, num_filters_total, 1, self.art_num_filters]
            art_b = tf.Variable(tf.constant(0.1, shape=[self.art_num_filters]))
            art_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="art_W")

            conv = tf.nn.conv2d(self.h_pool_flat, art_W, strides=[1, 1, 1, 1], padding="SAME")
            relu = tf.nn.relu(tf.nn.bias_add(conv, art_b))

            pooled = tf.reduce_max(relu, axis=1, keep_dims=True)
            art_pooled_outputs.append(pooled)

        art_num_filters_total = num_filters_total * self.art_num_filters * len(self.art_filter_sizes)

        # [1, art_num_filters]
        self.art_pool = tf.reshape(tf.concat(art_pooled_outputs, 3), [1, art_num_filters_total])

        # [batch_size, num_filters]
        self.local = tf.reshape(self.h_pool, [-1, num_filters_total])

        # context vector for each sentence
        # [batch_size, art_num_filters]
        self.context = tf.tile(self.art_pool, tf.reshape([self.batch_size, 1], [-1]))

        # [batch_size, num_filters + art_num_filters]
        self.sentence = tf.concat([self.local, self.context], 1)

        drop = tf.reshape(self.sentence, [-1, num_filters_total + art_num_filters_total])

        fc_drop = fully_connected(drop, 1, activation_fn=tf.sigmoid)

        # High rank sentences
        high_count = tf.ceil(tf.scalar_mul(0.2, tf.to_float(self.batch_size)))

        a = tf.reshape(fc_drop, [self.batch_size])
        b = tf.cast(high_count, tf.int32)

        self.highests = tf.nn.top_k(a, b)

        self.predictions = tf.reduce_mean(self.highests.values)

        self.predicted = tf.convert_to_tensor([[1 - self.predictions, self.predictions]])

        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                       logits=self.predicted)
        self.loss = tf.reduce_mean(self.xentropy)

        self.predicted_label = tf.argmax(self.predicted, 1)

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_label, self.output), tf.float32))

        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def run_model(self, batches, dev_batches, test_batches):
        init = tf.global_variables_initializer()
        with tf.Session() as self.sess:
            #init.run()
            self.sess.run(init)
            epoch = 1
            while True:
                ## Train
                epoch_loss = float(0)
                epoch += 1
                train_correct = 0
                train = 0
                for batch in batches:
                    X_batch, X_len, hate, _, _ = batch
                    if len(X_batch) == 1:
                        continue
                    feed_dict = {self.train_inputs: X_batch,
                                self.sequence_length: X_len,
                                self.keep_prob: self.keep_ratio,
                                self.output: [hate],
                                self.batch_size: len(X_batch)
                                 }

                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    loss_val, predictions_, _ = self.sess.run([self.loss, self.predicted_label, self.training_op], feed_dict= feed_dict)
                    train += 1
                    if predictions_ == hate:
                        train_correct += 1
                    epoch_loss += loss_val
                ## Test
                test_correct = 0
                test = 0
                y_true = list()
                y_pred = list()
                for X_batch, X_len, hate, _, _ in dev_batches:
                    feed_dict = {self.train_inputs: X_batch,
                                 self.sequence_length: X_len,
                                 self.keep_prob: 1,
                                 self.output: [hate],
                                 self.batch_size: len(X_batch)
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    try:
                        predictions_ = self.predicted_label.eval(feed_dict=feed_dict)
                        y_pred.append(predictions_)
                        y_true.append(hate)
                        if predictions_ == hate:
                            test_correct += 1
                        test += 1
                    except Exception:
                        print()
                print(epoch, "Train accuracy:", train_correct / train,
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", test_correct / test,
                      "F1: ", f1_score(y_true, y_pred))


                if epoch == self.epochs:
                    break
            y_pred = list()
            """
            for X_batch, X_len in test_batches:
                feed_dict = {self.train_inputs: X_batch,
                             self.sequence_length: X_len,
                             self.keep_prob: 1,
                             self.batch_size: len(X_batch)
                             }
                if self.pretrain:
                    feed_dict[self.embedding_placeholder] = self.my_embeddings
                try:
                    predictions_ = self.predicted_label.eval(feed_dict=feed_dict)
                    y_pred.append(predictions_)
                except Exception:
                    print()
            """
        return y_pred