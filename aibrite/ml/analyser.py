import concurrent.futures
import datetime
import os
import time
import uuid
from collections import namedtuple

from aibrite.ml.loggers import CsvLogger

import pandas as pd

from aibrite.ml.core import (MlBase, PredictionResult, TrainIteration,
                             TrainResult)
from aibrite.ml.neuralnet import NeuralNet

analyser_cache = {}

JobResult = namedtuple(
    'JobResult', 'train_time prediction_time id classifier hyper_parameters')


class AnalyserJob:

    def __init__(self, id, analyser, neuralnet, test_sets):
        self.id = id
        self.status = 'created'

        self.train_time = 0
        self.prediction_time = 0
        self.analyser = analyser
        self.neuralnet = neuralnet
        self.test_sets = test_sets

    def get_result(self):
        return JobResult(id=self.id,
                         train_time=self.train_time,
                         prediction_time=self.prediction_time,
                         classifier=self.neuralnet.__class__.__name__,
                         hyper_parameters=self.neuralnet.get_hyperparameters())

    def add_to_train_log(self, train_data, prediction=None, extra_data=None):
        return self.analyser.logger.add_to_train_log(self.neuralnet, train_data, prediction, extra_data)

    def add_to_prediction_log(self, test_set_id, prediction_result, extra_data=None):
        return self.analyser.logger.add_to_prediction_log(self.neuralnet, test_set_id, prediction_result, extra_data=None)


class NeuralNetAnalyser:

    def __init__(self, name, logger=None,  session_name=None, max_workers=None, executor=concurrent.futures.ThreadPoolExecutor, train_options=None, job_completed=None):
        self.name = name
        self.executor = executor(max_workers=max_workers)
        self.worker_list = []

        if logger is None:
            log_dir = os.path.join(
                './analyserlogs', CsvLogger.generate_file_name(name))
            logger = CsvLogger(self, base_dir=log_dir, overwrite=True)

        self.logger = logger

        self.logger.init()

        self.train_options = train_options if train_options != None else {
            'foo': 12
        }
        self.job_completed = job_completed
        if session_name is None:
            self.session_name = "Session {0:0>4}".format(
                self.logger.get_session_count() + 1)
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
        self.logger.add_to_classifier_instances(neuralnet)

        job = AnalyserJob(job_id, analyser, neuralnet, test_sets)

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
        self.logger.add_to_session_log()
        self.logger.done()

    def _complete_job(self, job_result):
        if self.job_completed != None:
            self.job_completed(self, job_result)
        self.job_results.append(job_result)

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
        prediction_log = self.logger.prediction_log

        current_pred_table = prediction_log[prediction_log['session_name']
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
                pred_totals[['test_set', 'f1', 'precision', 'recall', 'support',
                             'learning_rate', 'hidden_layers', 'iteration_count']])

        elapsed = (self.finish_time - self.start_time).total_seconds()
        print("\nCompleted at {0:.2f} seconds with max {1} workers.\n".format(elapsed,
                                                                              self.executor._max_workers))
