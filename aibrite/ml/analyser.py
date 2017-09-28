import concurrent.futures
import time
from aibrite.ml.core import TrainResult, MlBase, PredictionResult, TrainIteration
from aibrite.ml.neuralnet import NeuralNet
import uuid
import pandas as pd
import os
import datetime
import threading
from multiprocessing import Process, Lock


class AnalyserJob:
    def __init__(self, neuralnet, test_sets):
        self.neuralnet = neuralnet
        self.test_sets = test_sets
        self.id = str(uuid.uuid4())
        self.status = 'created'


class NeuralNetAnalyser:

    def save_logs(self):
        pred_file = os.path.join(self.log_dir, 'pred.csv')
        train_file = os.path.join(self.log_dir, 'train.csv')

        self.prediction_log.to_csv(pred_file)
        self.train_log.to_csv(train_file)

    def add_to_train_log(self, job, train_data, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        hyper_parameters = job.neuralnet.get_hyperparameters()
        now = datetime.datetime.now()

        base_cols = {
            'timestamp': now,
            'classifier': job.neuralnet.__class__.__name__,
            'classifier_id': job.neuralnet.instance_id
        }

        data = {**base_cols, **train_data, **hyper_parameters, **extra_data}

        self._trainlock.acquire()
        self.train_log = self.train_log.append(
            data, ignore_index=True)
        self._trainlock.release()

    def add_to_prediction_log(self, job, test_set_id, score, extra_data=None):
        extra_data = extra_data if extra_data != None else {}
        precision, recall, f1, support = score.totals
        hyper_parameters = job.neuralnet.get_hyperparameters()
        now = datetime.datetime.now()
        rows_to_add = []

        for i, v in enumerate(score.labels):
            base_cols = {
                'timestamp': now,
                'classifier': job.neuralnet.__class__.__name__,
                'classifier_id': job.neuralnet.instance_id,
                'test_set': test_set_id,
                'precision': score.precision[i],
                'recall': score.recall[i],
                'accuracy': score.accuracy,
                'f1': score.f1[i],
                'label': score.labels[i],
                'support': score.support[i],
                'job_id': job.id
            }

            data = {**base_cols, **hyper_parameters, **extra_data}
            rows_to_add.append(data)

        base_cols = {
            'timestamp': now,
            'classifier': job.neuralnet.__class__.__name__,
            'classifier_id': job.neuralnet.instance_id,
            'test_set': test_set_id,
            'precision': precision,
            'recall': recall,
            'accuracy': score.accuracy,
            'f1': f1,
            'support': support,
            'label': '__totals__',
            'job_id': job.id

        }

        data = {**base_cols, **hyper_parameters, **extra_data}
        rows_to_add.append(data)

        self._predlock.acquire()
        for item in rows_to_add:
            self.prediction_log = self.prediction_log.append(
                item, ignore_index=True)
        self._predlock.release()

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
        self._predlock = Lock()
        self._trainlock = Lock()

    def _train_callback(self, job, train_data):
        self.add_to_train_log(job, train_data._asdict())

    def _start(this, job_id):
        job = this.job_list[job_id]
        job.status = 'training:started'

        job.neuralnet.train(
            lambda neuralnet, train_data: this._train_callback(job, train_data))

        job.status = 'prediction:started'

        for test_set_id, test_set in job.test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = job.neuralnet.predict(test_set_x)
            score = NeuralNet.score_report(
                test_set_y, prediction_result.predicted)
            this.add_to_prediction_log(job, test_set_id, score)
            # print("{0}:\n{1}\n".format(
            #     job.neuralnet, NeuralNet.format_score_report(score)))

        job.status = 'completed'

        return job

    def submit(self, neuralnet, test_sets, **kvargs):

        job = AnalyserJob(neuralnet, test_sets)
        self.job_list[job.id] = job
        item = self.executor.submit(NeuralNetAnalyser._start, job.id)
        self.worker_list.append(item)

    def start(self):
        for future in self._as_completed():
            try:
                job = future.result()
            except Exception as exc:
                print(exc)
                self.worker_list.remove(future)
            else:
                pass
                print("job ok")

    def _as_completed(self):
        return concurrent.futures.as_completed(self.worker_list)

# def print_result(df):
#     df = df[['classifier', 'test_set', 'f1', 'iteration_count', 'hidden_layers', 'learning_rate']].sort_values(
#         ['f1'], ascending=False)
#     with pd.option_context('expand_frame_repr', False):
#         print(df)

    def print_summary(self):
        pred_totals = self.prediction_log[self.prediction_log['label']
                                          == '__totals__'].sort_values(['f1'], ascending=False)

        print("*" * 32)
        print("{:^32}".format("PREDICTION SUMMARY"))
        print("*" * 32)

        print("Predictions:")

        with pd.option_context('expand_frame_repr', False):
            print(pred_totals[['classifier', 'test_set', 'f1']])
