#!/usr/bin/env python
import os
import ssl

import gzip
import cPickle
from six.moves import urllib

cache_file = 'mnist.pkl.gz'
mnist_url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

# return data in pickle format
def load_mnist(cache_dir):
	cache_path = os.path.join(cache_dir, cache_file)

	# Download the MNIST dataset if it is not present
	if not os.path.isfile(cache_path):
		print('Downloading data from %s' % mnist_url)
		urllib.request.urlretrieve(mnist_url, cache_path)

	print('... loading data')

	# Load the cachepath
	with gzip.open(cache_path, 'rb') as f:
		try:
			dataset = cPickle.load(f, encoding='latin1')
		except:
			dataset = cPickle.load(f)

	return dataset

def parse_set(matrix_set):
	ins, outs = matrix_set
	print(ins.shape, outs.shape)

def parse_mnist(dataset):
	training_set, valid_set, test_set = (parse_set(dset) for dset in dataset)

mnist_dir = "/tmp"
dataset = load_mnist(mnist_dir)
parse_mnist(dataset)
