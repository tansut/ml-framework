import concurrent.futures
import datetime
import os
import time
import uuid
from collections import namedtuple
from threading import Lock
import re

import pandas as pd

from aibrite.ml.core import (MlBase, PredictionResult, TrainIteration,
                             TrainResult)
from aibrite.ml.neuralnet import NeuralNet

analyser_cache = {}

JobResult = namedtuple(
    'JobResult', 'train_data prediction_data train_time prediction_time id classifier hyper_parameters')


class AnalyserJob:

    def __init__(self, id, analyser, neuralnet, test_sets):
        self.id = id
        self.status = 'created'
        self._predlock = Lock()
        self._trainlock = Lock()
        self._prediction_data = []
        self._train_data = []
        self.train_time = 0
        self.prediction_time = 0
        self.analyser = analyser
        self.neuralnet = neuralnet
        self.test_sets = test_sets

    def get_result(self):
        return JobResult(train_data=self._train_data,
                         prediction_data=self._prediction_data,
                         id=self.id,
                         train_time=self.train_time,
                         prediction_time=self.prediction_time,
                         classifier=self.neuralnet.__class__.__name__,
                         hyper_parameters=self.neuralnet.get_hyperparameters())

    def add_to_train_log(self, train_data, prediction=None, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        hyper_parameters = self.neuralnet.get_hyperparameters()
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
            'classifier': self.neuralnet.__class__.__name__,
            'classifier_instance': self.id,
            'session_name': self.analyser.session_name
        }

        data = {**base_cols, **train_data, **hyper_parameters, **prediction_data, **extra_data}
        with self._trainlock:
            self._train_data.append(data)
        return data

    def add_to_prediction_log(self, test_set_id, prediction_result, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        score = prediction_result.score
        precision, recall, f1, support = score.totals
        hyper_parameters = self.neuralnet.get_hyperparameters()
        now = datetime.datetime.now()
        rows_to_add = []
        for i, v in enumerate(score.labels):
            base_cols = {
                'timestamp': now,
                'classifier': self.neuralnet.__class__.__name__,
                'test_set': test_set_id,
                'precision': score.precision[i],
                'recall': score.recall[i],
                'accuracy': score.accuracy,
                'f1': score.f1[i],
                'label': score.labels[i],
                'support': score.support[i],
                'classifier_instance': self.id,
                'prediction_time': prediction_result.elapsed(),
                'train_time': self.train_time,
                'session_name': self.analyser.session_name
            }

            data = {**base_cols, **hyper_parameters, **extra_data}
            rows_to_add.append(data)

        base_cols = {
            'timestamp': now,
            'classifier': self.neuralnet.__class__.__name__,
            'test_set': test_set_id,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support,
            'label': '__totals__',
            'classifier_instance': self.id,
            'prediction_time': prediction_result.elapsed(),
            'train_time': self.train_time,
            'session_name': self.analyser.session_name
        }

        data = {**base_cols, **hyper_parameters, **extra_data}
        rows_to_add.append(data)

        with self._predlock:
            for row in rows_to_add:
                self._prediction_data.append(row)
        return data


class NeuralNetAnalyser:

    #    def add_to_session_log(self, extra_data=None):
    #         extra_data = extra_data if extra_data != None else {}
    #         now = datetime.datetime.now()
    #         base_cols = {
    #             'timestamp': now,
    #             'session_name': self.session_name
    #         }
    #         data = {**base_cols, **extra_data}
    #         with self._session_file_lock:
    #             self._session_data.append(data)
    #         return data

    def save_logs(self):
        self.prediction_log.to_csv(self.pred_file, index=False)
        self.train_log.to_csv(self.train_file, index=False)
        self.session_log.to_csv(self.session_file, index=False)

    def _init_logs(self):

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

    def _append_job_data(self, train_data, prediction_data):
        for item in prediction_data:
            self.prediction_log = self.prediction_log.append(
                item, ignore_index=True)
        for item in train_data:
            self.train_log = self.train_log.append(
                item, ignore_index=True)

    def generate_file_name(self, s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

    def __init__(self, name, base_dir='./', overwrite=False, session_name=None, max_workers=None, executor=concurrent.futures.ProcessPoolExecutor, train_options=None, job_completed=None):
        self.name = name
        self.executor = executor(max_workers=max_workers)
        self.worker_list = []
        # self.job_list = {}

        self.base_dir = base_dir if base_dir != None else './'
        self.db_dir = os.path.join(
            self.base_dir, self.generate_file_name(name))

        self.pred_file = os.path.join(self.db_dir, 'pred.csv')
        self.train_file = os.path.join(self.db_dir, 'train.csv')
        self.session_file = os.path.join(self.db_dir, 'session.csv')

        self._session_data = []
        self._session_file_lock = Lock()
        self.overwrite = overwrite
        # self.use_subdir = use_subdir
        # if self.use_subdir:
        #     self.base_dir = os.path.join(
        # self.base_dir,
        # datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
        self._init_logs()
        self.train_options = train_options if train_options != None else {
            'foo': 12
        }
        self.job_completed = job_completed
        if session_name is None:
            self.session_name = "Session {0:0>4}".format(
                len(self.session_log) + 1)
        else:
            self.session_name = session_name
        analyser_cache[self.session_name] = self
        self.job_results = []
        self.job_counter = 1

    def _train_callback(self, job, neuralnet, train_iteration):
        test_sets = job.test_sets
        for test_set_id, test_set in test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = neuralnet.predict(
                test_set_x, expected=test_set_y)
            job.add_to_train_log(train_iteration._asdict(),
                                 prediction=(test_set_id, prediction_result))

    def _start_job(analyser_id, job_id, neuralnet_class, train_set, test_sets, **kvargs):
        analyser = analyser_cache[analyser_id]
        train_x, train_y = train_set
        neuralnet = neuralnet_class(train_x, train_y, **kvargs)

        job = AnalyserJob(job_id, analyser, neuralnet, test_sets)
        # analyser.job_list[job.id] = job

        job.status = 'training:started'
        neuralnet.train(lambda neuralnet, train_iteration: analyser._train_callback(
            job, neuralnet, train_iteration))
        job.train_time = neuralnet.train_result.elapsed()
        job.status = 'prediction:started'
        for test_set_id, test_set in test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = neuralnet.predict(
                test_set_x, expected=test_set_y)
            job.prediction_time += prediction_result.elapsed()
            job.add_to_prediction_log(test_set_id, prediction_result)

        job.status = 'completed'
        return job.get_result()

    def submit(self, neuralnet_class, train_set, test_sets, id=None, **kvargs):
        if id is None:
            id = "{0}/{1}".format(self.session_name, self.job_counter)
        else:
            id = id.format(self.job_counter)
        self.job_counter += 1
        item = self.executor.submit(
            NeuralNetAnalyser._start_job, self.session_name, id, neuralnet_class, train_set, test_sets, **kvargs)
        self.worker_list.append(item)

    def _complete_session(self):
        now = datetime.datetime.now()
        self.session_log = self.session_log.append({
            'timestamp': now,
            'session_name': self.session_name

        }, ignore_index=True)

        self.save_logs()

    def _complete_job(self, job_result):
        self._append_job_data(
            job_result.train_data, job_result.prediction_data)
        if self.job_completed != None:
            self.job_completed(self, job_result)
        self.job_results.append(job_result)
        self.save_logs()

    def join(self):
        self.start_time = datetime.datetime.now()
        for future in self._as_completed():
            try:
                job_result = future.result()
            except Exception as exc:
                print("ERROR")
                print(exc)
                raise exc
                self.worker_list.remove(future)
            else:
                self.worker_list.remove(future)
                self._complete_job(job_result)
                if len(self.worker_list) <= 0:
                    self._complete_session()
        self.finish_time = datetime.datetime.now()

    def _as_completed(self):
        return concurrent.futures.as_completed(self.worker_list)

    def print_summary(self):
        current_pred_table = self.prediction_log[self.prediction_log['session_name']
                                                 == self.session_name]
        pred_totals = current_pred_table[current_pred_table['label']
                                         == '__totals__'].sort_values(['f1'], ascending=False)
        title = "{0}/{1}".format(self.name, self.session_name)
        print("\n", "*" * 48)
        print("{:^48}".format(title.upper()))
        print("*" * 48, "\n")

        total_train_time = sum(r.train_time for r in self.job_results)
        total_prediction_time = sum(
            r.prediction_time for r in self.job_results)

        # print("Total train/prediction seconds: {0:.2f}/{1:.2f}\n".format(
        #     total_train_time, total_prediction_time))

        # print("Predictions:")

        with pd.option_context('expand_frame_repr', False):
            # for result in self.job_results:
            #     df = pred_totals[pred_totals['job_id'] == result.id]
            #     print("{0} {1}".format(result.classifier, result.train_time))
            print(
                pred_totals[['test_set', 'f1', 'precision', 'recall', 'support', 'learning_rate', 'hidden_layers', 'iteration_count']])

        elapsed = (self.finish_time - self.start_time).total_seconds()
        print("\nCompleted at {0:.2f} seconds with max {1} workers.\n".format(elapsed,
                                                                              self.executor._max_workers))
