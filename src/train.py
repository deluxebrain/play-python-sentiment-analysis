from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
import os
import re
import numpy as np
import nltk
import pickle

porter = PorterStemmer()
stop = stopwords.words('english')
work_path = os.path.join(os.path.expanduser('~'), 'tmp/datasets')


def tokenizer(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
        text = re.sub('[\W]+',
                      ' ',
                      text.lower()) + ' '.join(emoticons).replace('-', '')
        tokenized = [w for w in tokenizer_porter(text) if w not in stop]
        # tokenized = [w for w in text.split() if w not in stop]
        return tokenized


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer_simple(text):
    return text.split()


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv)  # skip header
        for line in csv:
            # [:-3] -> everything except the last 3 chars (review)
            # [-2]  -> last but 1 char (sentiment)
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def train():
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2**21,
                             preprocessor=None,
                             ngram_range=(1, 3),
                             tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    stream_path = os.path.join(work_path, 'movie_data.csv')
    doc_stream = stream_docs(path=stream_path)

    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % clf.score(X_test, y_test))

    clf = clf.partial_fit(X_test, y_test)

    return clf


def dump(object, name):
        dest = os.path.join(work_path, 'pkl_objects')
        if not os.path.exists(dest):
            os.makedirs(dest)

        file_name = name + '.pkl'
        pickle.dump(object,
                    open(os.path.join(dest, file_name), 'wb'),
                    protocol=2)


def main():

    nltk.download('stopwords')

    clf = train()

    dump(clf, 'classifier')
    dump(stop, 'stopwords')

main()
