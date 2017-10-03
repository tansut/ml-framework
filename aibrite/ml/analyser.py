import concurrent.futures
import datetime
import os
import time
import uuid
from collections import namedtuple, OrderedDict
import numpy as np
import sys
import numbers
import decimal

from aibrite.ml.loggers import CsvLogger, DefaultLgogger

import pandas as pd

from aibrite.ml.core import (MlBase, PredictionResult, TrainIteration,
                             TrainResult)
from aibrite.ml.neuralnet import NeuralNet

analyser_cache = {}

JobResult = namedtuple(
    'JobResult', 'train_result prediction_results prediction_totals id order classifier hyper_parameters')


class Change:
    def __init__(self, new, old, name):
        self.new = old
        self.old = new
        self.name = name
        if (isinstance(new, numbers.Number)):
            self.isNumeric = True
            self.change = 0.0 if new == 0 else (
                new - old) / old
        else:
            self.percent = 0
            self.isNumeric = False

    def formated_percent(self):
        if (self.isNumeric):
            return "{0:.2f}{1}".format(100 * self.change, self.change_symbol())
        else:
            return "-"

    def change_symbol(self):
        if not self.isNumeric:
            return ""
        if self.new > self.old:
            return chr(8595)
        elif self.new < self.old:
            return chr(8593)
        else:
            return ""


class ModelResult:
    def __init__(self, model_analyser, job_result):
        self.job_result = job_result
        self.model_analyser = model_analyser
        self._title = None

    def job_title(self):
        if self._title is not None:
            return self._title
        bests = []
        worsts = []
        best_values = []
        worst_values = []
        job_result = self.job_result
        for test_set_id, best in self.model_analyser.best_models.items():
            worst = self.model_analyser.worst_models[test_set_id]
            if (test_set_id != '__overall__'):
                pred_result = self.job_result.prediction_results[test_set_id]
                precision, recall, f1, support = pred_result.score.totals
                if (best.job_result.prediction_results[test_set_id] == pred_result):
                    best_values.append(f1)
                    bests.append(test_set_id)
                elif worst.job_result.prediction_results[test_set_id] == pred_result:
                    worsts.append(test_set_id)
                    worst_values.append(f1)
            else:
                f1 = self.job_result.prediction_totals[2]
                if (np.array_equal(best.job_result.prediction_totals, self.job_result.prediction_totals)):
                    best_values.append(f1)
                    bests.append(test_set_id)
                elif np.array_equal(worst.job_result.prediction_totals, self.job_result.prediction_totals):
                    worsts.append(test_set_id)
                    worst_values.append(f1)
        results = []
        if (len(bests) > 0):
            format = "{0}" if len(bests) == 1 else "{0}"
            results.append(
                "*BEST* on [" + ", ".join(bests) + "]:" + format.format(", ".join(["{0:.2f}".format(v) for v in best_values])))
        if (len(worsts) > 0):
            format = "{0}" if len(bests) == 1 else "{0}"
            results.append(
                "*WORST* on [" + ", ".join(worsts) + "]:" + format.format(", ".join(["{0:.2f}".format(v) for v in worst_values])))
        best_worst = ",".join(results)
        self._title = "{0}: {1}".format(job_result.id, best_worst) if len(
            results) > 0 else job_result.id
        return self._title

    def make_changes(self, ref_model_result):
        self.prediction_changes = {}
        overall_precision, overall_recall, overall_f1, overall_support = self.job_result.prediction_totals
        for best_test_set, best_mr in self.model_analyser.best_models.items():
            for test_set, result in self.job_result.prediction_results.items():
                precision, recall, f1, support = result.score.totals
                ref_precision, ref_recall, ref_f1, ref_support = best_mr.job_result.prediction_results[
                    test_set].score.totals
                change = self.prediction_changes[test_set, best_test_set] = Change(
                    f1, ref_f1, "f1")

        ref_overall_precision, ref_overall_recall, ref_overall_f1, ref_overall_support = best_mr.job_result.prediction_totals
        self.prediction_changes["__overall__", "__overall__"] = Change(
            overall_f1, ref_overall_f1, "f1")

        self.hyper_parameter_changes = self.get_dictinary_changes(ref_model_result.job_result.hyper_parameters,
                                                                  self.job_result.hyper_parameters)

    def get_dictinary_changes(self, ref, d):
        res = {}
        for k, v in d.items():
            if ref.get(k) is None:
                res[k] = Change(v, None, k)
            elif ref[k] != v:
                res[k] = Change(v, ref[k], k)
            else:
                pass
        for k, v in ref.items():
            if d.get(k) is None:
                res[k] = Change(None, v, k)
        return res


class ModelAnalyser:
    def __init__(self, analyser):
        self.analyser = analyser
        self.model_results = list(map(lambda jr: ModelResult(self, jr), sorted(
            analyser.job_results, key=lambda jr: jr.order, reverse=False)))

        self.test_set_names = list(set(sum([i for i in (list(mr.job_result.prediction_results.keys())
                                                        for mr in self.model_results)], [])))

        self.model_results_sorted = self.get_models_sorted()

        self.best_models = {k: v[-1]
                            for k, v in self.model_results_sorted.items()}

        self.worst_models = {k: v[0]
                             for k, v in self.model_results_sorted.items()}

    def generate_stats(self, target_job_result):
        for mr in self.model_results:
            mr.make_changes(target_job_result)

    def get_models_sorted(self, reverse=False):
        test_sets = self.test_set_names
        test_sets.append("__overall__")

        result = {}

        for ts in test_sets:
            if ts == "__overall__":
                result[ts] = sorted(
                    self.model_results, key=lambda mr: mr.job_result.prediction_totals[2], reverse=reverse)
            else:
                result[ts] = sorted(
                    self.model_results, key=lambda mr: mr.job_result.prediction_results[ts].score.totals[2], reverse=reverse)

        return result

    def print_subtitle(self, title):
        print("\n")
        print("{0:^80}".format("." * 40))
        print("{0:^80}".format(title))
        print("{0:^80}".format("." * 40))

    def print_h1(self, title):
        print("\n")
        print("*" * 80)
        print("{:^80}".format(title))
        print("*" * 80, "\n")

    def print_job(self, model_result):
        print("{0:<80}".format(model_result.job_title()))
        print("-" * 80)
        jr = model_result.job_result
        last_iteration = jr.train_result.last_iteration
        print(ModelAnalyser.format_dict(
            jr.hyper_parameters, use_cols=True))
        self.print_subtitle("Prediction performance")
        title_format = "{test_set:<16}{f1:>10}{precision:>10}{recall:>10}{support:>10}{time:>8}{change:>10}"
        title = title_format.format(
            test_set="", precision="prec", recall="recall", f1="f1", support="support", time="time", change="change%")

        print(title)
        for test_set, result in jr.prediction_results.items():
            precision, recall, f1, support = result.score.totals
            values_format = "{test_set:<16}{f1:10.2f}{precision:10.2f}{recall:10.2f}{support:>10}{time:8.2f}{change:>9}"

            print(values_format.format(
                test_set=test_set,
                f1=f1,
                precision=precision,
                recall=recall,
                support=support,
                time=result.elapsed,
                change=model_result.prediction_changes[test_set, test_set].formated_percent()))
        self.print_subtitle("Train performance")
        last_iteration = model_result.job_result.train_result.last_iteration
        print("{cost_title:<16}{maxcost:10.2f}".format(
            cost_title="cost",
            maxcost=last_iteration.max_cost))

        print("{train_title:<16}{train_time:10.2f}".format(
            train_title="train time",
            train_time=jr.train_result.elapsed))

    def print_summary(self, target=None):
        all_test_sets = self.test_set_names

        if (len(all_test_sets) == 1):
            target = all_test_sets[0]
        elif target is None:
            target = '__overall__'

        best_by_target = self.model_results_sorted[target][-1]
        worst_by_target = self.model_results_sorted[target][0]

        self.generate_stats(best_by_target)

        for mr in self.model_results:
            self.print_job(mr)
            print("-" * 80)
            print("\n")

        self.print_h1("Best Results")
        title_format = "{test_set:<12}{f1:>10}{precision:>10}{recall:>10}{support:>10}{model:>12}"
        title = title_format.format(
            test_set="", precision="prec", recall="recall", f1="f1", support="support", time="time", model="model")

        print(title)
        for test_set, mr in self.best_models.items():
            if (test_set == '__overall__'):
                precision, recall, f1, support = mr.job_result.prediction_totals
            else:
                pred_result = mr.job_result.prediction_results[test_set]
                precision, recall, f1, support = pred_result.score.totals
            values_format = "{test_set:<12}{f1:10.2f}{precision:10.2f}{recall:10.2f}{support:>10}{model:>12}"

            print(values_format.format(
                test_set=test_set,
                f1=f1,
                precision=precision,
                recall=recall,
                support=support,
                model=mr.job_result.id))

        self.print_h1("Best Models")

        single_hyper_parameter_models = list(filter(lambda mr: len(
            mr.hyper_parameter_changes) == 1, self.model_results))
        for mr in set(self.best_models.values()):
            self.print_job(mr)
            self.print_subtitle("hyper parameter tunings")
            title = "{0:<16}{1:>10}{2:>10}".format(
                "", "curr", "new")
            for k, v in mr.job_result.prediction_results.items():
                title += "{0:>12}".format(k + " f1")

            print(title)

            for mrs in single_hyper_parameter_models:
                # self.print_job(mrs)
                change_on_hp = list(mrs.hyper_parameter_changes.values())[0]
                if change_on_hp.isNumeric:
                    line = "{0:<16}{1:>10}{2:>10}".format(change_on_hp.name,
                                                          change_on_hp.new, change_on_hp.old)
                else:
                    line = "{0:<16}{1:>10}{2:>10}".format(change_on_hp.name,
                                                          str(change_on_hp.new), str(change_on_hp.old))
                used_combinations = {}
                for pred_key, pred_change in mrs.prediction_changes.items():
                    test_set, best_test_set = pred_key
                    if (mr != self.best_models[best_test_set] or used_combinations.get(test_set, None) != None):
                        continue
                    used_combinations[test_set] = self.best_models[best_test_set]
                    line += "{0:>12}".format(pred_change.formated_percent())
                print(line)
            print("-" * 80)
            print("\n")

        if (target != '__overall__'):
            print(
                "Here is the score report for {0}".format(best_by_target.job_result.id))
            score = best_by_target.job_result.prediction_results[target].score
            print(NeuralNet.format_score_report(score))

    def format_dict(d, use_cols=False):
        fmt_str = ""
        i = 0
        if (use_cols):
            for k, v in d.items():
                if i % 2 == 0 and i != 0 and i != len(d) - 1:
                    fmt_str += "\n"
                val = "{key:<20}:{value}".format(key=k, value=v)
                fmt_str += "{0:<40}".format(val)
                i += 1
        else:
            for k, v in d.items():
                val = "{key:<20}:{value}".format(key=k, value=v)
                fmt_str += val + ("\n" if i != len(d) - 1 else "")
                i += 1
        return fmt_str


class AnalyserJob:

    def __init__(self, id, order, analyser, neuralnet, test_sets):
        self.id = id
        self.order = order
        self.status = 'created'
        self.analyser = analyser
        self.neuralnet = neuralnet
        self.test_sets = test_sets
        self.prediction_results = {}
        self.prediction_totals = ()
        self.train_result = None

    def get_result(self):
        return JobResult(id=self.id,
                         order=self.order,
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

    def __init__(self, group=None, logger=None,  session_name=None, max_workers=None, executor=concurrent.futures.ThreadPoolExecutor, train_options=None, job_completed=None):
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

    def _start_job(analyser_id, job_id, job_order, neuralnet_class, train_set, test_sets, **kvargs):
        analyser = analyser_cache[analyser_id]
        train_x, train_y = train_set
        neuralnet = neuralnet_class(train_x, train_y, **kvargs)
        analyser.logger.add_to_classifier_instances(neuralnet)
        neuralnet.instance_id = job_id
        job = AnalyserJob(job_id, job_order, analyser, neuralnet, test_sets)

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
        support_total = pred_totals[3]
        job.prediction_totals = pred_totals / len(job.prediction_results)
        job.prediction_totals[3] = support_total.astype(int)
        job.status = 'completed'
        analyser.logger.flush()
        return job.get_result()

    def submit(self, neuralnet_class, train_set, test_sets, id=None, **kvargs):
        if id is None:
            id = "Model {0}".format(self.job_counter)
        else:
            id = id.format(self.job_counter)
        item = self.executor.submit(
            NeuralNetAnalyser._start_job, self.session_name, id, self.job_counter,  neuralnet_class, train_set, test_sets, **kvargs)
        self.job_counter += 1
        self.worker_list.append(item)

    def _complete_session(self):

        self.logger.update_session({
            'status': 'completed'
        })
        self.logger.done()
        self.model_analyser = ModelAnalyser(self)

    def _complete_job(self, job_result):
        if self.job_completed != None:
            self.job_completed(self, job_result)
        self.job_results.append(job_result)

    def join(self):
        print("Waiting for {0} models to run".format(
            self.job_counter), end='')
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

    def get_testset_from_user(self):
        test_set_indexes = {}
        i = 0
        test_set_indexes[str(i)] = '__overall__'
        print("{0}: {1}".format(0, 'Overall'))
        all_test_sets = self.model_analyser.test_set_names

        for ts in all_test_sets:
            i += 1
            print("{0}: {1}".format(i, ts))
            test_set_indexes[str(i)] = ts

        selected = input("\nWhich test set do you want to focus on ?")
        ts = test_set_indexes.get(selected, "__overall__")
        return ts

    def print_summary(self, target=None):
        self.model_analyser.print_summary(target)
