import concurrent.futures
import datetime
import os
import time
import uuid
from collections import namedtuple, OrderedDict
import numpy as np
import sys

from aibrite.ml.loggers import CsvLogger, DefaultLgogger

import pandas as pd

from aibrite.ml.core import (MlBase, PredictionResult, TrainIteration,
                             TrainResult)
from aibrite.ml.neuralnet import NeuralNet

analyser_cache = {}

JobResult = namedtuple(
    'JobResult', 'train_result prediction_results prediction_totals id classifier hyper_parameters')


class AnalyserJob:

    def __init__(self, id, analyser, neuralnet, test_sets):
        self.id = id
        self.status = 'created'
        self.analyser = analyser
        self.neuralnet = neuralnet
        self.test_sets = test_sets
        self.prediction_results = {}
        self.prediction_totals = ()
        self.train_result = None

    def get_result(self):
        return JobResult(id=self.id,
                         train_result=self.train_result,
                         prediction_results=self.prediction_results,
                         prediction_totals=self.prediction_totals,
                         classifier=self.neuralnet.__class__.__name__,
                         hyper_parameters=self.neuralnet.get_hyperparameters())

    def add_to_train_log(self, train_data, prediction=None, extra_data=None):
        return self.analyser.logger.add_to_train_log(self.neuralnet, train_data, prediction, extra_data)

    def add_to_prediction_log(self, test_set_id, prediction_result, extra_data=None):
        return self.analyser.logger.add_to_prediction_log(self.neuralnet, test_set_id, prediction_result, extra_data=None)


class NeuralNetAnalyser:

    def __init__(self, group=None, logger=None,  session_name=None, max_workers=None, executor=concurrent.futures.ProcessPoolExecutor, train_options=None, job_completed=None):
        group = group if group is not None else ''
        self.group = group
        self.executor = executor(max_workers=max_workers)
        self.worker_list = []
        self.logger = logger

        if logger is None:
            # log_dir = os.path.join(
            #     './analyserlogs', CsvLogger.generate_file_name(group))
            # logger = CsvLogger(self, base_dir=log_dir, overwrite=True)

            self.logger = DefaultLgogger(self)
            self.logger.init()

        else:
            # logger = logger(self, conn_str='mongodb://localhost:27017')
            # self.logger = logger
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
        pred_totals = np.asarray([0., 0., 0., 0])
        for test_set_id, test_set in test_sets.items():
            test_set_x, test_set_y = test_set
            prediction_result = neuralnet.predict(
                test_set_x, expected=test_set_y)
            job.prediction_results[test_set_id] = prediction_result
            job.add_to_prediction_log(test_set_id, prediction_result)
            pred_totals += np.asarray(prediction_result.score.totals)

        job.prediction_totals = pred_totals / len(job.prediction_results)
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
        print("Waiting for {0} jobs".format(self.job_counter), end='')
        sys.stdout.flush()
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
                print("{0}".format("."), end='')
                sys.stdout.flush()
                if len(self.worker_list) <= 0:
                    self._complete_session()
                    print("\n")

        self.finish_time = datetime.datetime.now()

    def _as_completed(self):
        return concurrent.futures.as_completed(self.worker_list)

    def format_dict(d):
        fmt_str = ""
        i = 0
        for k, v in d.items():
            if i % 2 == 0 and i != 0:
                fmt_str += "\n"
            val = "{key:<20}:{value}".format(key=k, value=v)
            fmt_str += "{0:<40}".format(val)
            i += 1
        return fmt_str

    def changed_dict_items(ref, d):
        res = {}
        for k, v in d.items():
            if ref.get(k) is None:
                res[k + ' *'] = v
            elif ref[k] != v:
                res[k + '*'] = "{1} ({0})".format(ref[k], v)
            else:
                res[k] = v
        for k, v in ref.items():
            if d.get(k) is None:
                res[k + '*'] = v
        return res

    def get_testset_from_user(self):
        test_set_indexes = {}
        i = 0
        test_set_indexes[str(i)] = '__totals__'
        print("{0}: {1}".format(0, 'ALL'))
        all_test_sets = self.get_unique_testset_names()

        for ts in all_test_sets:
            i += 1
            print("{0}: {1}".format(i, ts))
            test_set_indexes[str(i)] = ts

        selected = input("Which test set do you want to optimize ?")
        ts = test_set_indexes.get(selected, "__totals__")
        return ts

    def get_unique_testset_names(self):
        return list(set(sum([i for i in (list(jr.prediction_results.keys())
                                         for jr in self.job_results)], [])))

    def print_summary(self, target=None):

        all_test_sets = self.get_unique_testset_names()

        if (len(all_test_sets) == 1):
            target = all_test_sets[0]
        elif target is None:
            target = '__totals__'

        title = "{0}/{1}".format(self.group, self.session_name)
        print("\n")
        print("*" * 80)
        print("{:^80}".format(title.upper()))
        print("*" * 80, "\n")

        if (target == '__totals__'):
            def sort_fn(jr): return jr.prediction_totals[2]
        else:
            def sort_fn(
                jr): return jr.prediction_results[target].score.totals[2]
        job_results_sorted = sorted(
            self.job_results, key=sort_fn, reverse=False)
        prev_jr = None
        best = job_results_sorted[len(job_results_sorted) - 1]
        for jr in job_results_sorted:
            print("-" * 80)
            print("{0:^80}".format(jr.id.upper()))
            # print("-" * 48)
            # print("\n")
            last_iteration = jr.train_result.last_iteration
            print("-" * 80)

            if prev_jr is None:
                print(NeuralNetAnalyser.format_dict(jr.hyper_parameters))
            else:
                # print("changes based on {0}:\n".format(prev_jr.id))
                print(NeuralNetAnalyser.format_dict(NeuralNetAnalyser.changed_dict_items(
                    prev_jr.hyper_parameters, jr.hyper_parameters)))
            prev_jr = jr

            title = "{test_set:<10}{precision:>10}{recall:>10}{f1:>10}{support:>10}{time:>8}".format(
                test_set="set", precision="precision", recall="recall", f1="f1", support="support", time="time")
            print("\n")
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
            print("\n")
            # print("Train summary: costs (max/min/avg) {maxcost:.2f}/{mincost:.2f}/{avgcost:.2f}, traintime {train_time:.2f}".format(
            #     mincost=last_iteration.min_cost,
            #     maxcost=last_iteration.max_cost,
            #     avgcost=last_iteration.avg_cost,
            #     train_time=jr.train_result.elapsed))
        print("." * 80)
        print("{:^80}".format("Summary"))
        print("." * 80, "\n")
        print(
            "Best seems {0} (last one) based on [{1}] test set.".format(best.id, target))
        if (target != '__totals__'):
            print("Here is the score report.")
            best_score = best.prediction_results[target].score
            print(NeuralNet.format_score_report(best_score))
        elapsed = (self.finish_time - self.start_time).total_seconds()
        print("\nCompleted at {0:.2f} seconds with max {1} workers.\n".format(elapsed,
                                                                              self.executor._max_workers))
