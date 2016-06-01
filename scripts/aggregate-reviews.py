from subprocess import check_output, CalledProcessError
import pyprind
import pandas as pd
import os
import tempfile
import numpy as np 

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
                dataframe = dataframe.append([[txt, labels[sentiment]]], ignore_index=True)
                progress_bar.update()
    
    dataframe.columns = ['review', 'sentiment']
    
    np.random.seed(0)
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

    home_dir = os.path.join(os.path.expanduser('~'), 'tmp')
    temp_dir = tempfile.mkdtemp(dir=home_dir)
    dataframe_path = os.path.join(temp_dir, 'movie_data.csv')
    print ("Saving dataframe to {0}".format(dataframe_path))
    dataframe.to_csv(dataframe_path)

main()



