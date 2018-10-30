import os

from gensim.models import KeyedVectors

from title_classifier.model import TitleClassifier
from title_classifier.utils import create_data, is_title, is_not_title, standardize_item

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
_test_filename = os.path.join(_path, '../data/validation.csv')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))


def test(data, model):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for item in data:
        text_vectors = item['text_vectors']
        attributes = item['attributes']
        coordinates = item['coordinates']
        expected = standardize_item(item['is_title'])
        prediction = model.predict(text_vectors, attributes, coordinates)
        if prediction == expected and expected == is_title:
            true_positives += 1
        if prediction == expected and expected == is_not_title:
            true_negatives += 1
        if prediction != expected and expected == is_title:
            false_negatives += 1
        if prediction != expected and expected == is_not_title:
            false_positives += 1
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * 1 / (1 / precision + 1 / recall)
        print('precision', precision)
        print('recall', recall)
        print('f1', f1)
    except:
        print('Cannot compute precision and recall.')


if __name__ == '__main__':
    data = create_data(_test_filename, _model)
    first_model = 0
    last_model = 20
    for i in range(first_model, last_model + 1, 1):
        print('\nTesting model', i)
        nn_model = TitleClassifier.load(_saving_dir + '/title-classifier-' + str(i) + '.tf')
        test(data, nn_model)
