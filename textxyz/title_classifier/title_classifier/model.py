import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

TINY = 1e-6
ONE = tf.constant(1.)
NAMESPACE = 'title_classifier'
forbidden_weight = 1.
_weight_for_positive_matches = 1.
_rw = 1e-1


class TitleClassifier(object):
    _text_vocab_size = 50
    _text_vector_size = 50
    _word_proj_size_for_rnn = 30

    _attributes_input_size = 3
    _coordinates_input_size = 4
    _attributes_embedding_size = 15
    _coordinates_embedding_size = 15

    _internal_proj_size = 100
    _output_size = 2

    _memory_dim = 100
    _stack_dimension = 2

    def __init__(self):
        tf.reset_default_graph()
        with tf.variable_scope(NAMESPACE):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)

            # Input variables
            self.text_vectors_fw = tf.placeholder(tf.float32, shape=(None, None, self._text_vocab_size),
                                                  name='question_vectors_inp_fw')
            self.text_vectors_bw = tf.placeholder(tf.float32, shape=(None, None, self._text_vocab_size),
                                                  name='question_vectors_inp_nw')
            self.attributes = tf.placeholder(tf.float32, shape=(None, self._attributes_input_size),
                                             name='question_vectors_inp_nw')
            self.coordinates = tf.placeholder(tf.float32, shape=(None, self._coordinates_input_size),
                                              name='question_vectors_inp_nw')

            # The text is pre-processed by a bi-GRU
            self.Wq = tf.Variable(tf.random_uniform([self._text_vocab_size,
                                                     self._word_proj_size_for_rnn], -_rw, _rw))
            self.bq = tf.Variable(tf.random_uniform([self._word_proj_size_for_rnn], -_rw, _rw))
            self.internal_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wq) + self.bq)
            self.text_int_fw = tf.map_fn(self.internal_projection, self.text_vectors_fw)
            self.text_int_bw = tf.map_fn(self.internal_projection, self.text_vectors_bw)

            self.rnn_cell_fw = rnn.MultiRNNCell([rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                                                state_is_tuple=True)
            self.rnn_cell_bw = rnn.MultiRNNCell([rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                                                state_is_tuple=True)
            with tf.variable_scope('fw'):
                _, state_fw = tf.nn.dynamic_rnn(self.rnn_cell_fw, self.text_int_fw, time_major=True,
                                                dtype=tf.float32)
            with tf.variable_scope('bw'):
                _, state_bw = tf.nn.dynamic_rnn(self.rnn_cell_bw, self.text_int_bw, time_major=True,
                                                dtype=tf.float32)

            self.states = tf.concat(values=[state_fw[-1], state_bw[-1]], axis=1)
            self.Wqa = tf.Variable(
                tf.random_uniform([2 * self._memory_dim, self._text_vector_size], -_rw, _rw),
                name='Wqa')
            self.bqa = tf.Variable(tf.random_uniform([self._text_vector_size], -_rw, _rw), name='bqa')
            self.text_vector = tf.nn.relu(tf.matmul(self.states, self.Wqa) + self.bqa)

            # Dense layers for attributes and coordinate vectors
            self.Wa = tf.Variable(tf.random_uniform([self._attributes_input_size,
                                                     self._attributes_embedding_size], -_rw, _rw))
            self.ba = tf.Variable(tf.random_uniform([self._attributes_embedding_size], -_rw, _rw))
            self.attributes_embeddings = tf.nn.relu(tf.matmul(self.attributes, self.Wa) + self.ba)

            self.Wc = tf.Variable(tf.random_uniform([self._coordinates_input_size,
                                                     self._coordinates_embedding_size], -_rw, _rw))
            self.bc = tf.Variable(tf.random_uniform([self._coordinates_embedding_size], -_rw, _rw))
            self.coordinates_embeddings = tf.nn.relu(tf.matmul(self.coordinates, self.Wc) + self.bc)

            # Concatenating text_vector, attributes and coordinates
            self.concatenated = tf.concat(
                values=[self.text_vector, self.attributes_embeddings, self.coordinates_embeddings], axis=1)

            # Final feedforward layers
            self.Wh = tf.Variable(
                tf.random_uniform([self._text_vector_size
                                   + self._attributes_embedding_size
                                   + self._coordinates_embedding_size,
                                   self._internal_proj_size], -_rw, _rw),
                name='Ws1')
            self.bh = tf.Variable(tf.random_uniform([self._internal_proj_size], -_rw, _rw), name='bs1')
            self.hidden = tf.nn.relu(tf.matmul(self.concatenated, self.Wh) + self.bh)

            self.Wf = tf.Variable(tf.random_uniform([self._internal_proj_size, self._output_size], -_rw, _rw),
                                  name='Wf')
            self.bf = tf.Variable(tf.random_uniform([self._output_size], -_rw, _rw), name='bf')
            self.outputs = tf.nn.softmax(tf.matmul(self.hidden, self.Wf) + self.bf)

            # Loss function and training
            self.y_ = tf.placeholder(tf.float32, shape=(None, self._output_size), name='y_')
            self.one = tf.ones_like(self.outputs)
            self.tiny = self.one * TINY
            self.cross_entropy = (tf.reduce_mean(
                -tf.reduce_sum(self.y_ * tf.log(self.outputs + self.tiny) * _weight_for_positive_matches
                               + (self.one - self.y_) * tf.log(
                    self.one - self.outputs + self.tiny))
            ))

        # Clipping the gradient
        optimizer = tf.train.AdamOptimizer(1e-4)
        gvs = optimizer.compute_gradients(self.cross_entropy)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if var.name.find(NAMESPACE) != -1]
        self.train_step = optimizer.apply_gradients(capped_gvs)
        self.sess.run(tf.global_variables_initializer())

    def _add_identity(self, A):
        num_nodes = A.shape[0]
        identity = np.identity(num_nodes)
        return identity + A

    def __train(self, text_vectors, attributes, coordinates, y):
        text_vectors = np.array(text_vectors)
        text_vectors_fw = np.transpose(text_vectors, (1, 0, 2))
        text_vectors_bw = text_vectors_fw[::-1, :, :]

        attributes = np.array(attributes)
        coordinates = np.array(coordinates)

        y = np.array(y)

        feed_dict = {}
        feed_dict.update({self.text_vectors_fw: text_vectors_fw})
        feed_dict.update({self.text_vectors_bw: text_vectors_bw})
        feed_dict.update({self.attributes: attributes})
        feed_dict.update({self.coordinates: coordinates})
        feed_dict.update({self.y_: y})

        loss, _ = self.sess.run([self.cross_entropy, self.train_step], feed_dict)
        return loss

    def train(self, data, epochs=20):
        for epoch in range(epochs):
            loss = self.__train([data[i][0] for i in range(len(data))],
                                [data[i][1] for i in range(len(data))],
                                [data[i][2] for i in range(len(data))],
                                [data[i][3] for i in range(len(data))],
                                )
            print(loss)
            sys.stdout.flush()

    def __predict(self, text_vectors, attributes, coordinates):
        text_vectors = np.array(text_vectors)
        text_vectors_fw = np.transpose(text_vectors, (1, 0, 2))
        text_vectors_bw = text_vectors_fw[::-1, :, :]

        attributes = np.array(attributes)
        coordinates = np.array(coordinates)

        feed_dict = {}
        feed_dict.update({self.text_vectors_fw: text_vectors_fw})
        feed_dict.update({self.text_vectors_bw: text_vectors_bw})
        feed_dict.update({self.attributes: attributes})
        feed_dict.update({self.coordinates: coordinates})
        y_batch = self.sess.run([self.outputs], feed_dict)
        return y_batch

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, text_vectors, attributes, coordinates):
        output = self.__predict([text_vectors], [attributes], [coordinates])
        return self.__standardize_item(output[0][0])

    # Loading and saving functions

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    def load_tensorflow(self, filename):
        saver = tf.train.Saver([v for v in tf.global_variables() if NAMESPACE in v.name])
        saver.restore(self.sess, filename)

    @classmethod
    def load(self, filename):
        model = TitleClassifier()
        model.load_tensorflow(filename)
        return model
