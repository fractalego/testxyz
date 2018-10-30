import tensorflow.keras as keras

_sentence_vocab_size = 50
_word_proj_size_for_rnn = 25
_memory_dim = 100
_stack_dimension = 1

_attributes_input_size = 3
_coordinates_input_size = 4
_attributes_embedding_size = 15
_coordinates_embedding_size = 15

_hidden_dim = 200

_output_size = 2


def get_keras_model(dropout):
    rnn_model = keras.models.Sequential([
        keras.layers.Dense(_word_proj_size_for_rnn, activation='relu', input_shape=(None, _sentence_vocab_size)),
        keras.layers.Bidirectional(keras.layers.GRU(_memory_dim)),
        keras.layers.Dense(_hidden_dim, activation='relu'),
    ])

    attributes_model = keras.models.Sequential([
        keras.layers.Dense(_word_proj_size_for_rnn, activation='relu', input_shape=(_attributes_input_size,)),
        keras.layers.Dense(_attributes_embedding_size, activation='relu'),
    ])

    coordinates_model = keras.models.Sequential([
        keras.layers.Dense(_word_proj_size_for_rnn, activation='relu', input_shape=(_coordinates_input_size,)),
        keras.layers.Dense(_coordinates_embedding_size, activation='relu'),
    ])

    merged = keras.layers.Concatenate(axis=1)([rnn_model.output, attributes_model.output, coordinates_model.output])
    merged = keras.layers.Dense(_hidden_dim, activation='relu')(merged)
    merged = keras.layers.Dropout(dropout)(merged)
    merged = keras.layers.Dense(_output_size, activation='softmax')(merged)

    model = keras.models.Model(inputs=[rnn_model.input, attributes_model.input, coordinates_model.input],
                               outputs=merged)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    model = get_keras_model(dropout=0.7)
