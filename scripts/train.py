from subprocess import check_output, CalledProcessError
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
import pandas as pd
import os
# import tempfile
import re
import numpy as np
import nltk
import pickle

porter = PorterStemmer()
stop = stopwords.words('english')
work_path = os.path.join(os.path.expanduser('~'), 'tmp/datasets')


def preprocessor(text):
        # remove html
        text = re.sub('<[^>]*>', '', text)

        # save emoticons
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

        # remove non-word characters and put emoticons back
        text = re.sub('[\W]+',
                      ' ',
                      text.lower()) + ' '.join(emoticons).replace('-', '')

        return text


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
    with open(path, 'r', encoding='utf-8') as csv:
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


def aggregate_datasets():
    progress_bar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}

    dataframe = pd.DataFrame()

    # aggregate datasets to single csv
    for dataset in ('test', 'train'):
        for sentiment in ('pos', 'neg'):

            subpath = 'datasets/aclImdb/%s/%s' % (dataset, sentiment)
            path = os.path.join(git_root(), subpath)
            for file in os.listdir(path):
                with open(os.path.join(path, file),
                          'r', encoding='utf-8') as infile:
                    txt = infile.read()
                dataframe = dataframe.append([[txt, labels[sentiment]]],
                                             ignore_index=True)
                progress_bar.update()

    dataframe.columns = ['review', 'sentiment']

    return dataframe


def randomize_dataframe(dataframe):
    np.random.seed(0)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    return dataframe


def save_dataframe(dataframe):
    # home_dir = os.path.join(os.path.expanduser('~'), 'tmp')
    # temp_dir = tempfile.mkdtemp(dir=home_dir)
    dataframe_path = os.path.join(work_path, 'movie_data.csv')
    print("Saving dataframe to {0}".format(dataframe_path))
    dataframe.to_csv(dataframe_path, index=False)

    return dataframe_path


def load_dataframe(path):
    dataframe = pd.read_csv(path, index=False)
    return dataframe


def train(dataframe):
    x_train = dataframe.loc[:25000, 'review'].values
    y_train = dataframe.loc[:25000, 'sentiment'].values
    x_test = dataframe.loc[25000:, 'review'].values
    y_test = dataframe.loc[25000:, 'sentiment'].values

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer_simple, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer_simple, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]

    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=5,
                               verbose=1,
                               n_jobs=-1)

    gs_lr_tfidf.fit(x_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(x_test, y_test))


def dump(clf):
        dest = os.path.join(work_path, 'pkl_objects')
        if not os.path.exists(dest):
            os.makedirs(dest)

        pickle.dump(stop,
                    open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
                    protocol=4)
        pickle.dump(clf,
                    open(os.path.join(dest, 'classifier.pkl'), 'wb'),
                    protocol=4)


def main():

    nltk.download('stopwords')

    # df = aggregate_datasets()
    # df = randomize_dataframe(df)

    # preprocess each review in the dataset
    # df['review'] = df['review'].apply(preprocessor)

    # save_dataframe(df)

    # train(df)

    #
    # equivalent to above but using streamed dataset
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

    dump(clf)

main()
