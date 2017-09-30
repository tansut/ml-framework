import os
import datetime
import re
from threading import Lock
import pandas as pd
from pymongo import MongoClient


class AnalyserLoggerBase:
    def __init__(self, analyser):
        self.analyser = analyser

    def init(self):
        pass

    def done(self):
        pass

    def add_to_train_log(self, neuralnet, train_data, prediction=None, extra_data=None):
        pass

    def add_to_prediction_log(self, neuralnet, test_set_id, prediction_result, extra_data=None):
        pass

    def update_session(self, values):
        pass

    def get_session_count(self):
        pass

    def add_to_classifier_instances(self, neuralnet):
        pass

    def create_session(self):
        pass


class MongodbLogger(AnalyserLoggerBase):

    def __init__(self, conn_str='mongodb://localhost:27017'):
        super().__init__()
        self.conn_str = conn_str
        self.client = MongoClient(conn_str)

        self.db = self.client.nn_sandbox
        self.collections = {
            'session': self.db.session,
            'train': self.db.train,
            'prediction': self.db.prediction
        }

    def create_session():
        analyser = self.analyser
        data = {
            'session_name': analyser.session_name,
            'group_name': analyser.group,
            'timestamp': datetime.datetime.now(),
            'status': 'created'
        }

        try:
            inserted_session = self.collections['session'].insert_one(data)
            self.session_id = inserted_session.inserted_id
            return self.session_id
        except Exception as e:
            print(str(e.args))

    def add_to_classifier_instances(self, neuralnet):
        # data = {
        #     ''
        # }
        pass


class CsvLogger(AnalyserLoggerBase):

    def get_session_count(self):
        return len(self.session_log)

    def add_to_classifier_instances(self, neuralnet):
        pass

    def done(self):
        for item in self._prediction_data:
            self.prediction_log = self.prediction_log.append(
                item, ignore_index=True)
        for item in self._train_data:
            self.train_log = self.train_log.append(
                item, ignore_index=True)

        self.prediction_log.to_csv(self.pred_file, index=False)
        self.train_log.to_csv(self.train_file, index=False)
        self.session_log.to_csv(self.session_file, index=False)

    def init(self):

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        if os.path.exists(self.pred_file) and not self.overwrite:
            self.prediction_log = pd.read_csv(self.pred_file)
        else:
            self.prediction_log = pd.DataFrame(columns=[
                'session_name', 'classifier', 'test_set', 'label', 'f1', 'precision', 'recall', 'accuracy', 'support'])

        if os.path.exists(self.train_file) and not self.overwrite:
            self.train_log = pd.read_csv(self.train_file)
        else:
            self.train_log = pd.DataFrame(columns=[
                'session_name', 'classifier', 'cost'])
        if os.path.exists(self.session_file) and not self.overwrite:
            self.session_log = pd.read_csv(self.session_file)
        else:
            self.session_log = pd.DataFrame()

    def __init__(self, analyser, base_dir='./', overwrite=False):
        super().__init__(analyser)
        self.base_dir = base_dir if base_dir != None else './'
        self.db_dir = self.base_dir
        self.overwrite = overwrite

        self.pred_file = os.path.join(self.db_dir, 'pred.csv')
        self.train_file = os.path.join(self.db_dir, 'train.csv')
        self.session_file = os.path.join(self.db_dir, 'session.csv')

        self._predlock = Lock()
        self._trainlock = Lock()
        self._prediction_data = []
        self._train_data = []

    def generate_file_name(s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

    def create_session(self):
        extra_data = {}
        now = datetime.datetime.now()
        base_cols = {
            'timestamp': now,
            'session_name': self.analyser.session_name,
            'group_name': self.analyser.group
        }
        data = {**base_cols, **extra_data}
        self.session_log = self.session_log.append(data, ignore_index=True)

    def add_to_train_log(self, neuralnet, train_data, prediction=None, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        hyper_parameters = neuralnet.get_hyperparameters()
        now = datetime.datetime.now()
        if prediction is None:
            prediction_data = {}
        else:
            test_set_id, prediction_result = prediction
            precision, recall, f1, support = prediction_result.score.totals
            prediction_data = {
                'test_set': test_set_id,
                'precision': precision,
                'recall': recall,
                'accuracy': prediction_result.score.accuracy,
                'f1': f1,
                'support': support,
            }
        base_cols = {
            'timestamp': now,
            'classifier': neuralnet.__class__.__name__,
            'classifier_instance': neuralnet.instance_id,
            'session_name': self.analyser.session_name
        }

        data = {**base_cols, **train_data, **
                hyper_parameters, **prediction_data, **extra_data}

        with self._trainlock:
            self._train_data.append(data)
        return data

    def add_to_prediction_log(self, neuralnet, test_set_id, prediction_result, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        score = prediction_result.score
        precision, recall, f1, support = score.totals
        hyper_parameters = neuralnet.get_hyperparameters()
        now = datetime.datetime.now()
        rows_to_add = []
        for i, v in enumerate(score.labels):
            base_cols = {
                'timestamp': now,
                'classifier': neuralnet.__class__.__name__,
                'test_set': test_set_id,
                'precision': score.precision[i],
                'recall': score.recall[i],
                'accuracy': score.accuracy,
                'f1': score.f1[i],
                'label': score.labels[i],
                'support': score.support[i],
                'classifier_instance': neuralnet.instance_id,
                'prediction_time': prediction_result.elapsed(),
                'train_time': neuralnet.train_result.elapsed(),
                'session_name': self.analyser.session_name
            }

            data = {**base_cols, **hyper_parameters, **extra_data}
            rows_to_add.append(data)

        base_cols = {
            'timestamp': now,
            'classifier': neuralnet.__class__.__name__,
            'test_set': test_set_id,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support,
            'label': '__totals__',
            'classifier_instance': neuralnet.instance_id,
            'prediction_time': prediction_result.elapsed(),
            'train_time': neuralnet.train_result.elapsed(),
            'session_name': self.analyser.session_name
        }

        data = {**base_cols, **hyper_parameters, **extra_data}
        rows_to_add.append(data)

        with self._predlock:
            for row in rows_to_add:
                self._prediction_data.append(row)
        return data
