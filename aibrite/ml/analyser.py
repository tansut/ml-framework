import concurrent.futures
import time
from aibrite.ml.core import TrainResult, MlBase, PredictionResult, TrainIteration
from aibrite.ml.neuralnet import NeuralNet
import uuid
import pandas as pd
import os
import datetime
from threading import Lock
from collections import namedtuple

analyser_cache = {}

JobResult = namedtuple(
    'JobResult', 'train_data prediction_data train_time prediction_time id classifier hyper_parameters')


class AnalyserJob:

    def __init__(self, classifier, hyper_parameters):
        self.id = str(uuid.uuid4())
        self.status = 'created'
        self._predlock = Lock()
        self._trainlock = Lock()
        self._prediction_data = []
        self._train_data = []
        self.train_time = 0
        self.prediction_time = 0
        self.classifier = classifier
        self.hyper_parameters = hyper_parameters

    def get_result(self):
        return JobResult(train_data=self._train_data,
                         prediction_data=self._prediction_data,
                         id=self.id,
                         train_time=self.train_time,
                         prediction_time=self.prediction_time,
                         classifier=self.classifier,
                         hyper_parameters=self.hyper_parameters)

    def add_to_train_log(self, neuralnet, train_data, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        hyper_parameters = neuralnet.get_hyperparameters()
        now = datetime.datetime.now()

        base_cols = {
            'timestamp': now,
            'classifier': neuralnet.__class__.__name__,
            # 'classifier_id': neuralnet.instance_id,
            'job_id': self.id
        }

        data = {**base_cols, **train_data, **hyper_parameters, **extra_data}
        with self._trainlock:
            self._train_data.append(data)
        return data

    def add_to_prediction_log(self, neuralnet, test_set_id, score, elapsed, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        precision, recall, f1, support = score.totals
        hyper_parameters = neuralnet.get_hyperparameters()
        now = datetime.datetime.now()

        for i, v in enumerate(score.labels):
            base_cols = {
                'timestamp': now,
                'classifier': neuralnet.__class__.__name__,
                # 'classifier_id': neuralnet.instance_id,
                'test_set': test_set_id,
                'precision': score.precision[i],
                'recall': score.recall[i],
                'accuracy': score.accuracy,
                'f1': score.f1[i],
                'label': score.labels[i],
                'support': score.support[i],
                'job_id': self.id,
                'prediction_time': elapsed,
                'train_time': self.train_time
            }

            data = {**base_cols, **hyper_parameters, **extra_data}

        base_cols = {
            'timestamp': now,
            'classifier': neuralnet.__class__.__name__,
            # 'classifier_id': neuralnet.instance_id,
            'test_set': test_set_id,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support,
            'label': '__totals__',
            'job_id': self.id,
            'prediction_time': elapsed,
            'train_time': self.train_time
        }

        data = {**base_cols, **hyper_parameters, **extra_data}
        with self._predlock:
            self._prediction_data.append(data)
        return data


class NeuralNetAnalyser:

    def save_logs(self):
        pred_file = os.path.join(self.log_dir, 'pred.csv')
        train_file = os.path.join(self.log_dir, 'train.csv')

        self.prediction_log.to_csv(pred_file)
        self.train_log.to_csv(train_file)

    def _init_logs(self):
        self.prediction_log = pd.DataFrame(columns=[
            'timestamp', 'classifier', 'test_set', 'label', 'f1', 'precision', 'recall', 'accuracy', 'support'])

        self.train_log = pd.DataFrame(columns=[
            'timestamp', 'classifier', 'cost', 'epoch'])

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _append_job_data(self, train_data, prediction_data):
        for item in prediction_data:
            self.prediction_log = self.prediction_log.append(
                item, ignore_index=True)
        for item in train_data:
            self.train_log = self.train_log.append(
                item, ignore_index=True)

    def __init__(self, log_dir='./', use_subdir=True, max_workers=None, executor=concurrent.futures.ProcessPoolExecutor, train_options=None, job_completed=None):
        self.executor = executor(max_workers=max_workers)
        self.worker_list = []
        self.job_list = {}

        self.log_dir = log_dir if log_dir != None else './'
        self.use_subdir = use_subdir
        if self.use_subdir:
            self.log_dir = os.path.join(
                self.log_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
        self._init_logs()
        self.train_options = train_options if train_options != None else {
            'foo': 12
        }
        self.job_completed = job_completed
        self.id = str(uuid.uuid4())
        analyser_cache[self.id] = self
        self.job_results = []

    def _start_job(analyser_id, neuralnet_class, train_set, test_sets, **kvargs):
        analyser = analyser_cache[analyser_id]
        train_x, train_y = train_set
        neuralnet = neuralnet_class(train_x, train_y, **kvargs)

        job = AnalyserJob(neuralnet_class.__name__,
                          neuralnet.get_hyperparameters())
        analyser.job_list[job.id] = job

        job.status = 'training:started'
        neuralnet.train(lambda neuralnet, train_data: job.add_to_train_log(
            neuralnet, train_data._asdict()))
        job.train_time = neuralnet.train_result.elapsed()
        job.status = 'prediction:started'
        for test_set_id, test_set in test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = neuralnet.predict(test_set_x)
            elapsed = prediction_result.elapsed()
            job.prediction_time += elapsed
            score = NeuralNet.score_report(
                test_set_y, prediction_result.predicted, labels=neuralnet.labels)
            job.add_to_prediction_log(neuralnet, test_set_id, score, elapsed)
            # print("{0}:\n{1}\n".format(
            #     job.neuralnet, NeuralNet.format_score_report(score)))

        job.status = 'completed'
        return job.get_result()

    def submit(self, neuralnet_class, train_set, test_sets, **kvargs):
        item = self.executor.submit(
            NeuralNetAnalyser._start_job, self.id, neuralnet_class, train_set, test_sets, **kvargs)
        self.worker_list.append(item)

    def join(self):
        self.start_time = datetime.datetime.now()
        for future in self._as_completed():
            try:
                job_result = future.result()
            except Exception as exc:
                print("ERROR")
                print(exc)
                # raise exc
                self.worker_list.remove(future)
            else:
                self._append_job_data(
                    job_result.train_data, job_result.prediction_data)
                if self.job_completed != None:
                    self.job_completed(self, job_result)
                self.job_results.append(job_result)
                self.save_logs()
        self.finish_time = datetime.datetime.now()

    def _as_completed(self):
        return concurrent.futures.as_completed(self.worker_list)

    def print_summary(self):
        pred_totals = self.prediction_log[self.prediction_log['label']
                                          == '__totals__'].sort_values(['f1'], ascending=False)

        print("\n", "*" * 32)
        print("{:^32}".format("PREDICTION SUMMARY"))
        print("*" * 32, "\n\n")

        total_train_time = sum(r.train_time for r in self.job_results)
        total_prediction_time = sum(
            r.prediction_time for r in self.job_results)

        print("Total train/prediction seconds: {0:.2f}/{1:.2f}\n".format(
            total_train_time, total_prediction_time))

        # print("Predictions:")

        with pd.option_context('expand_frame_repr', False):
            # for result in self.job_results:
            #     df = pred_totals[pred_totals['job_id'] == result.id]
            #     print("{0} {1}".format(result.classifier, result.train_time))
            print(
                pred_totals[['test_set', 'f1', 'precision', 'recall', 'support', 'prediction_time', 'train_time']])

        elapsed = (self.finish_time - self.start_time).total_seconds()
        print("\nCompleted at {0:.2f} seconds with max {1} workers.\n".format(elapsed,
                                                                              self.executor._max_workers))
