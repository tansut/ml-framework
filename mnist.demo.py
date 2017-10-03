import six.moves.cPickle as pickle
import gzip
import os
import numpy as np

from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam
from aibrite.ml.neuralnet import NeuralNet


def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    return train_set, valid_set, test_set


def get_datasets():
    train_set, valid_set, test_set = load_data('./data/mnist.pkl.gz')

    train_x = train_set[0]
    train_y = train_set[1]

    test_x = test_set[0]
    test_y = test_set[1]

    valid_x = valid_set[0]
    valid_y = valid_set[1]

    return (train_x, train_y), (test_x, test_y), (valid_x, valid_y)


(train_x, train_y), (test_x, test_y), (valid_x, valid_y) = get_datasets()

nn = NeuralNetWithAdam(train_x, train_y,
                       hidden_layers=(9,),
                       iteration_count=100,
                       learning_rate=0.001,
                       minibatch_size=0,
                       epochs=1)
print("Training ...")
train_result = nn.train()

prediction_result = nn.predict(test_x)

report = NeuralNet.score_report(test_y, prediction_result.predicted)

print("{0}:\n{1}\n".format(
    nn, NeuralNet.format_score(report)))
