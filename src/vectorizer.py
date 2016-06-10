from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer
import re
import os
import pickle

work_path = os.path.join(os.path.expanduser('~'), 'tmp/datasets')

stop = pickle.load(open(
    os.path.join(work_path,
                 'pkl_objects',
                 'stopwords.pkl'),
    'rb'))

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         ngram_range=(1, 3),
                         tokenizer=tokenizer)
