import concurrent.futures
import time
from aibrite.ml.core import TrainResult, MlBase, PredictionResult, TrainIteration
from aibrite.ml.neuralnet import NeuralNet
import uuid
import pandas as pd
import os
import datetime
import multiprocessing
from multiprocessing import Process, Lock


class AnalyserJob:
    def __init__(self, neuralnet_class, train_set, test_sets, **args):
        self.neuralnet_class = neuralnet_class
        self.train_set = train_set
        self.test_sets = test_sets
        self.id = str(uuid.uuid4())
        self.status = 'created'
        self.args = args
        self.neuralnet = None


analyser_cache = {}
_joblist_lock = Lock()
_predlock = Lock()
_prediction_data = []


class NeuralNetAnalyser:

    def save_logs(self):
        print("Length of pred data (save)", self.id, len(_prediction_data))

        for item in _prediction_data:
            self.prediction_log = self.prediction_log.append(
                item, ignore_index=True)
        pred_file = os.path.join(self.log_dir, 'pred.csv')
        train_file = os.path.join(self.log_dir, 'train.csv')

        self.prediction_log.to_csv(pred_file)
        self.train_log.to_csv(train_file)

    def add_to_train_log(self, neuralnet, train_data, extra_data=None):
        pass
        # extra_data = extra_data if extra_data != None else {}
        # hyper_parameters = neuralnet.get_hyperparameters()
        # now = datetime.datetime.now()

        # base_cols = {
        #     'timestamp': now,
        #     'classifier': neuralnet.__class__.__name__,
        #     'classifier_id': neuralnet.instance_id
        # }

        # data = {**base_cols, **train_data, **hyper_parameters, **extra_data}

        # self._trainlock.acquire()
        # self.train_log = self.train_log.append(
        #     data, ignore_index=True)
        # self._trainlock.release()

    def add_to_prediction_log(self, neuralnet, test_set_id, score, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        precision, recall, f1, support = score.totals
        hyper_parameters = neuralnet.get_hyperparameters()
        now = datetime.datetime.now()
        rows_to_add = []

        for i, v in enumerate(score.labels):
            base_cols = {
                'timestamp': now,
                'classifier': neuralnet.__class__.__name__,
                'classifier_id': neuralnet.instance_id,
                'test_set': test_set_id,
                'precision': score.precision[i],
                'recall': score.recall[i],
                'accuracy': score.accuracy,
                'f1': score.f1[i],
                'label': score.labels[i],
                'support': score.support[i],
                'job_id': 1
            }

            data = {**base_cols, **hyper_parameters, **extra_data}
            rows_to_add.append(data)

        base_cols = {
            'timestamp': now,
            'classifier': neuralnet.__class__.__name__,
            'classifier_id': neuralnet.instance_id,
            'test_set': test_set_id,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support,
            'label': '__totals__',
            'job_id': 1

        }

        data = {**base_cols, **hyper_parameters, **extra_data}

        # rows_to_add.append(data)
        # print("About to write to prediclog", len(rows_to_add))
        _predlock.acquire()
        _prediction_data.append(data)
        print("Length of pred data (add)", self.id, len(_prediction_data))

        # for item in rows_to_add:
        #     self.prediction_log = self.prediction_log.append(
        #         item, ignore_index=True)
        #     print(self.prediction_log)
        _predlock.release()

    def _init_logs(self):
        self.prediction_log = pd.DataFrame(columns=[
            'timestamp', 'classifier', 'classifier_id', 'test_set', 'label', 'f1', 'precision', 'recall', 'accuracy', 'support'])

        self.train_log = pd.DataFrame(columns=[
            'timestamp', 'classifier', 'classifier_id', 'cost', 'epoch', 'current_minibatch_index'])

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def __init__(self, log_dir='./', labels=None, max_workers=None, executor=concurrent.futures.ProcessPoolExecutor, train_options=None):
        self.executor = executor(max_workers=None)
        self.worker_list = []
        self.job_list = {}
        self.log_dir = log_dir if log_dir != None else './'
        self.log_dir = os.path.join(
            self.log_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
        self.labels = labels
        self._init_logs()
        self.train_options = train_options if train_options != None else {
            'foo': 12
        }
        self.id = str(uuid.uuid4())
        analyser_cache[self.id] = self
        self._predlock = Lock()
        self._trainlock = Lock()
        self._joblist_lock = Lock()
        # self._prediction_data = []

    def _train_callback(self, neuralnet, train_data):
        self.add_to_train_log(neuralnet, train_data._asdict())

    def _start_job(analyser_id, job_id):
        analyser = analyser_cache[analyser_id]
        _joblist_lock.acquire()
        job = analyser.job_list[job_id]
        _joblist_lock.release()
        print("jobs")
        print(analyser_cache[analyser_id].job_list.keys())
        analyser = analyser_cache[analyser_id]
        train_x, train_y = job.train_set
        neuralnet = job.neuralnet_class(train_x, train_y, **job.args)

        job.status = 'training:started'
        print("train starting")
        neuralnet.train()
        # lambda neuralnet, train_data: analyser._train_callback(neuralnet, train_data)
        job.status = 'prediction:started'
        print("test set count", len(job.test_sets))
        for test_set_id, test_set in job.test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = neuralnet.predict(test_set_x)
            score = NeuralNet.score_report(
                test_set_y, prediction_result.predicted)
            analyser.add_to_prediction_log(neuralnet, test_set_id, score)
            # print("{0}:\n{1}\n".format(
            #     job.neuralnet, NeuralNet.format_score_report(score)))

        job.status = 'completed'
        analyser.save_logs()
        return analyser_id

    def submit(self, neuralnet_class, train_set, test_sets, **kvargs):
        job = AnalyserJob(neuralnet_class, train_set, test_sets, **kvargs)
        _joblist_lock.acquire()
        self.job_list[job.id] = job
        print(job.id, " added")
        _joblist_lock.release()
        item = self.executor.submit(
            NeuralNetAnalyser._start_job, self.id, job.id)
        self.worker_list.append(item)

    def start(self):
        # for job_id, job in self.job_list.items():
        #     item = self.executor.submit(
        #         NeuralNetAnalyser._start_job, self.id, job_id)
        #     self.worker_list.append(item)
        for future in self._as_completed():
            try:
                analyser_id = future.result()
            except Exception as exc:
                print("ERROR")
                print(exc)
                # raise exc
                self.worker_list.remove(future)
            else:
                pass
                self.save_logs()
                analyser_cache[analyser_id]
                print(analyser_cache[analyser_id] == self)

    def _as_completed(self):
        return concurrent.futures.as_completed(self.worker_list)

    def print_summary(self):
        pred_totals = self.prediction_log[self.prediction_log['label']
                                          == '__totals__'].sort_values(['f1'], ascending=False)

        print("*" * 32)
        print("{:^32}".format("PREDICTION SUMMARY"))
        print("*" * 32)

        print("Predictions:")

        with pd.option_context('expand_frame_repr', False):
            print(pred_totals[['classifier', 'test_set', 'f1']])
