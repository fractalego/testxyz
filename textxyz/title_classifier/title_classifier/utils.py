import csv
import nltk
import numpy as np

from gensim import utils

is_title = [1, 0]
is_not_title = [0, 1]


def create_data(filename, model):
    data = []
    with open(filename, encoding="ISO-8859-1") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item = {}
            item['text_vectors'] = _create_vector_list_from_text(model, row['Text'])
            item['attributes'] = _create_vector_from_boolean_list([row['IsBold'],
                                                                   row['IsItalic'],
                                                                   row['IsUnderlined'],
                                                                   ])
            item['coordinates'] = _create_vector_from_coordinate_list([row['Left'],
                                                                       row['Right'],
                                                                       row['Top'],
                                                                       row['Bottom'],
                                                                       ])
            label = row['Label']
            item['is_title'] = is_title if label == '1' else is_not_title
            data.append(item)
    return data


def create_data_for_prediction_only(filename, model):
    data = []
    with open(filename, encoding="ISO-8859-1") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item = {}
            item['text_vectors'] = _create_vector_list_from_text(model, row['Text'])
            item['attributes'] = _create_vector_from_boolean_list([row['IsBold'],
                                                                   row['IsItalic'],
                                                                   row['IsUnderlined'],
                                                                   ])
            item['coordinates'] = _create_vector_from_coordinate_list([row['Left'],
                                                                       row['Right'],
                                                                       row['Top'],
                                                                       row['Bottom'],
                                                                       ])
            data.append(item)
    return data


def _create_vector_from_coordinate_list(lst):
    return_vect = []
    for item in lst:
        return_vect.append(float(item))
    return return_vect


def _create_vector_from_boolean_list(lst):
    return_vect = []
    for item in lst:
        if item == 'TRUE':
            return_vect.append(1.)
        else:
            return_vect.append(0.)
    return return_vect


def get_words(text):
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(utils.to_unicode(text))
    return words


def capitalize(word):
    return word[0].upper() + word[1:]


def low_case(word):
    return word.lower()


def infer_vector_from_word(model, word):
    vector = np.zeros(50)
    try:
        vector = model[word]
    except:
        try:
            vector = model[capitalize(word)]
        except:
            try:
                vector = model[low_case(word)]
            except:
                pass
    return vector


def _create_vector_list_from_text(model, text):
    words = get_words(text)
    vectors = []
    for word in words:
        vectors.append(infer_vector_from_word(model, word))
    return vectors


def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        seq_length = len(item['text_vectors'])
        try:
            size_to_data_dict[seq_length].append(item)
        except:
            size_to_data_dict[seq_length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets


def standardize_item(item):
    if item[0] < item[1]:
        return [0., 1.]
    return [1., 0.]
