import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict

from aibrite.ml.core import MlBase


class Analyser:
    def __init__(self):
        df = self.df = pd.DataFrame(columns=[
            'timestamp', 'classifier', 'test_set', 'f1', 'precision', 'recall', 'accuracy', 'support'])

    def to_csv(self, file_name):
        self.df.to_csv(file_name)

    def add_data(self, classifier, test_set, expected, predicted, hyper_parameters, labels=None, extra_data=None):
        score = MlBase.score_report(expected, predicted)
        precision, recall, f1, support = score.totals
        base_cols = {
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

        self.df = self.df.append(data, ignore_index=True)

        return self.df
