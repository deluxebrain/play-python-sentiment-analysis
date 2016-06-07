import pickle
import os
import numpy as np
import sys
from vectorizer import vect

work_path = os.path.join(os.path.expanduser('~'),
                         'tmp/datasets')

clf = pickle.load(open(os.path.join(work_path,
                                    'pkl_objects/classifier.pkl'),
                       'rb'))


def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba


def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

document = sys.stdin.readline()
label, proba = classify(document)

print("{0} with a probability of {1}".format(label, proba))
