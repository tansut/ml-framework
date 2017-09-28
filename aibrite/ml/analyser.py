import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict

from aibrite.ml.core import MlBase


class Analyser:
    def __init__(self):
        self.prediction_log = pd.DataFrame(columns=[
            'timestamp', 'classifier', 'test_set', 'f1', 'precision', 'recall', 'accuracy', 'support'])

        self.train_log = DataFrame(columns='timestamp', 'classifier', 'test_set', 'cost', 'epoch', 'current_minibatch_index'))

    def to_csv(self, file_name):
        self.prediction_log.to_csv(file_name)

    def add_to_train_log(self, classifier, instance_id, test_set, cost, epoch, current_minibatch_index, total_minibatch_index, extra_data = None, options = None):
        pass

    def add_to_prediction_log(self, classifier, instance_id, test_set, expected, predicted, hyper_parameters, labels = None, extra_data = None):
        score=MlBase.score_report(expected, predicted)
        precision, recall, f1, support=score.totals
        base_cols={
            'timestamp': datetime.datetime.now(),
            'classifier': classifier,
            'test_set': test_set,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support
        }

        data = OrderedDict({**base_cols, **hyper_parameters, **extra_data})

        self.prediction_log = self.prediction_log.append(
            data, ignore_index = True)

        return self.prediction_log
