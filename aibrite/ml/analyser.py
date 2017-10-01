import concurrent.futures
import datetime
import os
import time
import uuid
from collections import namedtuple, OrderedDict

from aibrite.ml.loggers import CsvLogger

import pandas as pd

from aibrite.ml.core import (MlBase, PredictionResult, TrainIteration,
                             TrainResult)
from aibrite.ml.neuralnet import NeuralNet

analyser_cache = {}

JobResult = namedtuple(
    'JobResult', 'train_result prediction_results id classifier hyper_parameters')


class AnalyserJob:

    def __init__(self, id, analyser, neuralnet, test_sets):
        self.id = id
        self.status = 'created'
        self.analyser = analyser
        self.neuralnet = neuralnet
        self.test_sets = test_sets
        self.prediction_results = {}
        self.train_result = None

    def get_result(self):
        return JobResult(id=self.id,
                         train_result=self.train_result,
                         prediction_results=self.prediction_results,
                         classifier=self.neuralnet.__class__.__name__,
                         hyper_parameters=self.neuralnet.get_hyperparameters())

    def add_to_train_log(self, train_data, prediction=None, extra_data=None):
        return self.analyser.logger.add_to_train_log(self.neuralnet, train_data, prediction, extra_data)

    def add_to_prediction_log(self, test_set_id, prediction_result, extra_data=None):
        return self.analyser.logger.add_to_prediction_log(self.neuralnet, test_set_id, prediction_result, extra_data=None)


class NeuralNetAnalyser:

    def __init__(self, group=None, logger=None,  session_name=None, max_workers=None, executor=concurrent.futures.ThreadPoolExecutor, train_options=None, job_completed=None):
        group = group if group is not None else ''
        self.group = group
        self.executor = executor(max_workers=max_workers)
        self.worker_list = []

        if logger is None:
            log_dir = os.path.join(
                './analyserlogs', CsvLogger.generate_file_name(group))
            logger = CsvLogger(self, base_dir=log_dir, overwrite=True)

            self.logger = logger
            self.logger.init()

        else:
            logger = logger(self, conn_str='mongodb://localhost:27017')
            self.logger = logger
            logger.init()

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

        self.logger.create_session()

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
        analyser.logger.add_to_classifier_instances(neuralnet)
        neuralnet.instance_id = job_id
        job = AnalyserJob(job_id, analyser, neuralnet, test_sets)

        job.status = 'training:started'
        neuralnet.train(lambda neuralnet, train_iteration: analyser._train_callback(
            job, neuralnet, train_iteration))
        job.train_result = neuralnet.train_result
        job.status = 'prediction:started'
        for test_set_id, test_set in test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = neuralnet.predict(
                test_set_x, expected=test_set_y)
            job.prediction_results[test_set_id] = prediction_result
            job.add_to_prediction_log(test_set_id, prediction_result)

        job.status = 'completed'
        analyser.logger.flush()
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
        self.logger.update_session({
            'status': 'completed'
        })
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

    def format_dict(d):
        fmt_str = ""
        for k, v in d.items():
            fmt_str += "{key:<20}:{value}\n".format(key=k, value=v)
        return fmt_str

    def changed_dict_items(ref, d):
        res = {}
        for k, v in d.items():
            if ref.get(k) is None:
                res[k] = v
            elif ref[k] != v:
                res[k] = "{0}->{1}".format(ref[k], v)
        for k, v in ref.items():
            if d.get(k) is None:
                res[k] = v
        return res

    def print_summary(self):
        title = "{0}/{1}".format(self.group, self.session_name)
        print("\n")
        print("*" * 48)
        print("{:^48}".format(title.upper()))
        print("*" * 48, "\n")

        prev_jr = None
        for jr in self.job_results:
            print("-" * 48)
            print("{0:^36}\n".format(jr.id.upper()))
            # print("-" * 48)
            # print("\n")
            last_iteration = jr.train_result.last_iteration

            # props = {
            #     'cost (max/min/avg)': "{maxcost:.2f}/{mincost:.2f}/{avgcost:.2f}".format(
            #         mincost=last_iteration.min_cost,
            #         maxcost=last_iteration.max_cost,
            #         avgcost=last_iteration.avg_cost),
            #     'train time': jr.train_result.elapsed
            # }
            # print(NeuralNetAnalyser.format_dict(props))
            print("-" * 48)

            tracked_items = OrderedDict({**props, **jr.hyper_parameters})
            if prev_jr is None:
                print(NeuralNetAnalyser.format_dict(jr.hyper_parameters))
                prev_jr = jr
            else:
                print("changes based on {0}:\n".format(prev_jr.id))
                print(NeuralNetAnalyser.format_dict(NeuralNetAnalyser.changed_dict_items(
                    prev_jr.hyper_parameters, jr.hyper_parameters)))

            title = "{test_set:<10}{precision:>10}{recall:>10}{f1:>10}{support:>10}{time:>8}".format(
                test_set="set", precision="precision", recall="recall", f1="f1", support="support", time="time")
            print(title)
            for test_set, result in jr.prediction_results.items():
                precision, recall, f1, support = result.score.totals
                print("{test_set:<10}{precision:10.2f}{recall:10.2f}{f1:10.2f}{support:>10}{time:8.2f}".format(
                    test_set=test_set,
                    f1=f1,
                    precision=precision,
                    recall=recall,
                    support=support,
                    time=result.elapsed))
            # print("\n")

        elapsed = (self.finish_time - self.start_time).total_seconds()
        print("\nCompleted at {0:.2f} seconds with max {1} workers.\n".format(elapsed,
                                                                              self.executor._max_workers))
