import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from log_reg import LogisticRegression, LearningAlgorithm

data_frame = pd.read_csv('winequality.txt', sep=';')
data = data_frame.values

train_ratio = 0.60
test_ratio = 0.2
validation_ratio = 0.2

mldata = LearningAlgorithm.split(
    data, train_ratio, test_ratio, validation_ratio)

train_x = mldata[0][:, 0:data.shape[1] - 1]


test_x = mldata[1][:, 0:data.shape[1] - 1]
validation_x = mldata[2][:, 0:data.shape[1] - 1]

train_y = mldata[0][:, data.shape[1] - 1:data.shape[1]]
test_y = mldata[1][:, data.shape[1] - 1:data.shape[1]]
validation_y = mldata[2][:, data.shape[1] - 1:data.shape[1]]

lrs = [0.0022]
iterations = [16000]

for it in iterations:
    for i, lr in enumerate(lrs):
        lra = LogisticRegression(train_x, train_y, lr=lr)
        train_result = lra.train(iterate=it)
        testResult = lra.test(
            train_result['w'], train_result['b'], test_x, test_y)
        print('it: {0} lr: {1}, success: {2:.2f}, last cost: {3:.5f}'.format(
            it, lr, (testResult['success'] / testResult['total']) * 100, np.sum(train_result['costs'][-1])))

# LearningAlgorithm.plotCosts(train_result["costs"])

exit()
