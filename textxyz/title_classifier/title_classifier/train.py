import os

from gensim.models import KeyedVectors

from title_classifier.utils import create_data, bin_data_into_buckets
from title_classifier.model import TitleClassifier

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save/')
_training_filename = os.path.join(_path, '../data/train.csv')
#_training_filename = os.path.join(_path, '../data/train_small.csv')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))


def train(data, model, saving_dir, name_prefix, epochs=20, bucket_size=10, trace_every=1):
    import random
    import sys

    buckets = bin_data_into_buckets(data, bucket_size)
    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        for bucket in random_buckets:
            graph_bucket = []
            try:
                for item in bucket:
                    text_vectors = item['text_vectors']
                    attributes = item['attributes']
                    coordinates = item['coordinates']
                    y = item['is_title']
                    graph_bucket.append((text_vectors, attributes, coordinates, y))
                if len(graph_bucket) > 0:
                    model.train(graph_bucket, 1)
            except Exception as e:
                print('Exception caught during training: ' + str(e))
        if i % trace_every == 0:
            save_filename = saving_dir + name_prefix + '-' + str(i) + '.tf'
            sys.stderr.write('Saving into ' + save_filename + '\n')
            model.save(save_filename)


if __name__ == '__main__':
    data = create_data(_training_filename, _model)
    nn_model = TitleClassifier()
    train(data,
          nn_model,
          _saving_dir,
          name_prefix='title-classifier',
          epochs=30,
          bucket_size=10,
          trace_every=1,
          )
