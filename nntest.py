import concurrent.futures
import time


class NNTester:
    def test_it(self, id, nn_class, train_x, train_y, test_x, test_y, **kvargs):
        nn = nn_class(train_x, train_y, **kvargs)
        train_result = nn.train()
        pred_result = nn.predict_and_test(test_x, test_y)
        return {
            'self': self
            'id': id,
            'pred_result': pred_result,
            'train_result': train_result,
            'nn': nn
        }

    def submit(self, id, nn_class, train_x, train_y, test_x, test_y, **kvargs):
        item = self.executor.submit(lambda id, nn_class, train_x, train_y, test_x, test_y,
                                    **kvargs: self.test_it(id, nn_class, train_x, train_y, test_x, test_y, **kvargs),
                                    id, nn_class, train_x, train_y, test_x, test_y, **kvargs)
        self.worker_list.append(item)
        return item

    def __init__(self):
        self.executor = None
        self.worker_list = []

    def as_completed(self):
        return concurrent.futures.as_completed(self.worker_list)

    def getExecutor(self, max_workers=None):
        if self.executor == None:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers)
        return self.executor
