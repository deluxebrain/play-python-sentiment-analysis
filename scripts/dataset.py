from subprocess import check_output, CalledProcessError
import pyprind
import pandas as pd
import os
import numpy as np

work_path = os.path.join(os.path.expanduser('~'), 'tmp/datasets')


def git_root():
    try:
        path = check_output(['git', 'rev-parse', '--show-toplevel'])
    except CalledProcessError:
        raise IOError('Current working directory is not a git repository')
    return path.decode('utf-8').strip()


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
    dataframe_path = os.path.join(work_path, 'movie_data.csv')
    print("Saving dataframe to {0}".format(dataframe_path))
    dataframe.to_csv(dataframe_path, index=False)

    return dataframe_path


df = aggregate_datasets()
df = randomize_dataframe(df)
save_dataframe(df)
