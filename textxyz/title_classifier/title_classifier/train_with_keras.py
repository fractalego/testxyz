import os

from gensim.models import KeyedVectors

from title_classifier.data_generator import DataGenerator
from title_classifier.keras_model import get_keras_model
from title_classifier.utils import create_data

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save/')
_training_filename = os.path.join(_path, '../data/train.csv')
# _training_filename = os.path.join(_path, '../data/train_small.csv')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))


def train(model, generator, saving_dir, epochs=20, trace_every=None):
    import sys

    for i in range(epochs):
        if trace_every:
            sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')

        model.fit_generator(generator=generator, epochs=1, verbose=1)
        if trace_every and i % trace_every == 0:
            save_filename = saving_dir + '/title-keras-' + str(i) + '.tf'
            sys.stderr.write('Saving into ' + save_filename + '\n')
            model.save(save_filename)


if __name__ == '__main__':
    data = create_data(_training_filename, _model)
    generator = DataGenerator(data, batch_size=20)
    nn_model = get_keras_model(dropout=0.7)

    train(nn_model, generator, _saving_dir, epochs=39, trace_every=1)
