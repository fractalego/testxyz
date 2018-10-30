import os
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from nlp_challenge.utils import get_texts, get_texts_and_labels
from nlp_challenge.validate import count_true_and_false_positives_and_negatives

_bucket_size = 10
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save/')
_training_filename = os.path.join(_path, '../data/train.csv')
_test_filename = os.path.join(_path, '../data/dev.csv')


class TfidfTransformer(object):
    _stemmer = nltk.PorterStemmer()

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize,
                                          stop_words=set(stopwords.words('english')),
                                          max_features=max_features)

    def save(self, filename):
        joblib.dump(self.vectorizer, filename)

    @staticmethod
    def load(filename):
        transformer = TfidfTransformer()
        transformer.vectorizer = joblib.load(filename)
        return transformer

    def _stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def _tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        stems = self._stem_tokens(tokens, self._stemmer)
        return stems


def train_and_save_tfidf_transformer():
    texts = get_texts(_training_filename)
    transformer = TfidfTransformer()
    transformer.vectorizer.fit(texts)
    transformer.save(os.path.join(_path, '../data/save/tfidf.save'))


def train_and_save_tree_classifier(depth):
    texts, labels = get_texts_and_labels(_training_filename)
    transformer = TfidfTransformer.load('../data/save/tfidf.save')
    cls = DecisionTreeClassifier(max_depth=depth)
    cls.fit(transformer.vectorizer.transform(texts), labels)
    joblib.dump(cls, os.path.join(_path, '../data/save/tree_cls.save'))


def _test(texts, labels, cls, transformer):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for text, label in zip(texts, labels):
        expected = label
        prediction = cls.predict(transformer.vectorizer.transform([text]))[0]
        new_true_positives, new_false_positives, new_true_negatives, new_false_negatives = \
            count_true_and_false_positives_and_negatives(prediction, expected)
        true_positives += new_true_positives
        false_positives += new_false_positives
        true_negatives += new_true_negatives
        false_negatives += new_false_negatives

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 / (1 / precision + 1 / recall)
        print('precision', precision)
        print('recall', recall)
        print('f1', f1)

    except:
        print('Cannot compute accuracy.')


def validate_tree_classifier():
    cls = joblib.load(os.path.join(_path, '../data/save/tree_cls.save'))
    texts, labels = get_texts_and_labels(_test_filename)
    transformer = TfidfTransformer.load(os.path.join(_path, '../data/save/tfidf.save'))
    _test(texts, labels, cls, transformer)


if __name__ == '__main__':
    for depth in range(10, 11):
        print('depth=', depth)
        #train_and_save_tree_classifier(depth=depth)
        validate_tree_classifier()
