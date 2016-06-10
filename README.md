# Performing sentiment analysis on a IMDB movie review dataset

## Pre-requisistes

* Movie review dataset

	```shell
	# wget:
	# -q	    : quiet
	# -O -	    : write to stdout
	# tar
	# v	    : verbose
	# x	    : extract the files
	# z	    : filter the archive through gzip
	# -	    : read from stdin
	# -C <dir>  : write to specified directory
	wget -q -O - http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz | tar vxz - -C /var/datasets/
	```

	I usually write large datasets to the ```/var``` volume to keep them out of the git repository.
	
	I'll then symlink it back in for convenience:
	
	```shell
	ln -s /var/datasets/aclImdb datasets/
	```

## Running the app

```shell
python app.py <<<"Some string to analyze"

```

