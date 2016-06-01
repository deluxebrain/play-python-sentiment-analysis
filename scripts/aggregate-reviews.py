from subprocess import check_output, CalledProcessError
import pyprind
import pandas as pd
import os

def git_root():
    try:
        path = check_output(['git', 'rev-parse', '--show-toplevel'])
    except CalledProcessError:
        raise IOError('Current working directory is not a git repository')
    return path.decode('utf-8').strip()

def main():
    progress_bar = pyprind.ProgBar(50000)
    labels = {'pos': 1, 'neg': 0}
    dataframe = pd.DataFrame()

    for dataset in ('test', 'train'):
        for sentiment in ('pos', 'neg'):
            path = os.path.join(git_root(), 'datasets/aclImdb/%s/%s' % (dataset, sentiment))
            for file in os.listdir(path):
               with open(os.path.join(path, file), 'r') as infile:
                  txt = infile.read()
             dataframe = dataframe.append([[txt, labels[1]]], ignore_index=True)
                progress_bar.update()
    dataframe.colums = ['review', 'sentiment']

main()



