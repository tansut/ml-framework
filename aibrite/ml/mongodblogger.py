from aibrite.ml.loggers import AnalyserLoggerBase

import os
import datetime
import re
from threading import Lock
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId


class MongodbLogger(AnalyserLoggerBase):

    def __init__(self, analyser, conn_str='mongodb://localhost:27017'):
        super().__init__(analyser)
        self.conn_str = conn_str
        self.client = MongoClient(conn_str)

        self.db = self.client.nn_sandbox
        self.collections = {
            'session': self.db.session,
            'train': self.db.train,
            'prediction': self.db.prediction,
            'classifier': self.db.classifier
        }

    def create_session(self):
        analyser = self.analyser
        data = {
            'session_name': analyser.session_name,
            'group_name': analyser.group,
            'timestamp': datetime.datetime.now(),
            'status': 'created'
        }

        try:
            inserted_session = self.collections.session.insert_one(data)
            self.session_id = inserted_session.inserted_id
            return self.session_id
        except Exception as e:
            print(str(e.args))

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

        data = {
            'timestamp': now,
            'session_id': self.session_id,
            'session_name': self.analyser.session_name,
            'classifier': neuralnet.__class__.__name__,
            'classifier_instance': neuralnet.instance_id,
            'extra_data': extra_data,
            'train_data': train_data,
            'prediction': prediction_data
        }

        try:
            inserted_train = self.collections.session.insert_one(data)
            return inserted_train.inserted_id
        except Exception as e:
            print(str(e.args))

    def add_to_classifier_instances(self, neuralnet):
        hyper_parameters = neuralnet.get_hyperparameters()
        data = {
            'session': self.session_id,
            'classifier': neuralnet.__class__.__name__,
            'classifier_instance': neuralnet.instance_id,
            'hyperparameters': hyper_parameters
        }
        try:
            inserted_classifier = self.collections.classifier.insert_one(data)
            return inserted_classifier.inserted_id
        except Exception as e:
            print(str(e.args))

    def add_to_prediction_log(self, neuralnet, test_set_id, prediction_result, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        score = prediction_result.score
        precision, recall, f1, support = score.totals
        hyper_parameters = neuralnet.get_hyperparameters()
        now = datetime.datetime.now()

        data = {
            'timestamp': now,
            'classifier': neuralnet.__class__.__name__,
            'test_set': test_set_id,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support,
            'label': '__overall__',
            'classifier_instance': neuralnet.instance_id,
            'prediction_time': prediction_result.elapsed,
            'train_time': neuralnet.train_result.elapsed,
            'session_name': self.analyser.session_name
        }

        try:
            inserted_prediction = self.collections.prediction.insert_one(data)
            return inserted_prediction.inserted_id
        except Exception as e:
            print(str(e.args))

    def update_session(self, values):
        # status = values.status
        # update_value = {
        #     'status': status
        # }
        try:
            self.collections.session.update_one({
                '_id': self.session_id,
                '$set': values
            })
        except Exception as e:
            print(str(e.args))

    def init(self):
        pass

    def done(self):
        pass
