import os
import sys

from gensim.models import KeyedVectors

from title_classifier.model import TitleClassifier
from title_classifier.utils import is_title
from title_classifier.utils import create_data_for_prediction_only

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))


_error_message = '''
Please provide a filename as an input.
You can the filename as an argument: python -m title_classifier.classify data/test.csv
'''


def get_prediction_list(data, model):
    return_list = []
    for item in data:
        text_vectors = item['text_vectors']
        attributes = item['attributes']
        coordinates = item['coordinates']
        prediction = model.predict(text_vectors, attributes, coordinates)
        if prediction == is_title:
            return_list.append(True)
        else:
            return_list.append(False)
    return return_list

if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        data = create_data_for_prediction_only(filename, _model)
        nn_model = TitleClassifier.load(_saving_dir + '/title-classifier-17.tf')
        prediction_list = get_prediction_list(data, nn_model)
        try:
            lines = open(filename, encoding="ISO-8859-1").read().split('\n')[1:]
        except Exception as e:
            print(str(e))
        for line, prediction in zip(lines, prediction_list):
            print(line, '=>', prediction)
    else:
        if os.isatty(0):
            print(_error_message)
            exit(0)



