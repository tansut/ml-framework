# Aibrite Machine Learning Framework

Neural network classes + neural network testing.

## Getting Started
```python
from aibrite.ml.neuralnet import NeuralNet
import pandas as pd

df = pd.read_csv("./data/ex2data1.csv")
train_set, dev_set, test_set = NeuralNet.split(df.values, 0.8, 0.1, 0.1)
train_x, train_y = train_set[:, 0:-1], train_set[:, -1]

nn = NeuralNet(train_x, train_y, hidden_layers=(2, 2), iteration_count=6000)
nn.train()
result = nn.predict(test_x, expected=test_y)

print("{0}:\n{1}\n".format(
    nn, NeuralNet.format_score(result.score)))
```
Output is:
```none
NeuralNet[it=6000,lr=0.0100,hl=(2, 2),lrd=0.0000,lambd=0.0001,batch=0,epochs=1, shuffle=False]:
label      precision    recall        f1   support
0.0             1.00      1.00      1.00         4
1.0             1.00      1.00      1.00         6
avg/total       1.00      1.00      1.00        10
Accuracy:  1.00
```

## Neuralnet Classes
```python
from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam
from aibrite.ml.neuralnetwithmomentum import NeuralNetWithMomentum
from aibrite.ml.neuralnetwithrmsprop import NeuralNetWithRMSprop


nn = NeuralNet(train_x, train_y, hidden_layers=(2, 2), iteration_count=6000)

nn = NeuralNetWithAdam(train_x, train_y, hidden_layers=(2, 2), iteration_count=6000, beta1=0.9, beta2=0.99)

nn = NeuralNetWithMomentum(train_x, train_y, hidden_layers=(2, 2), iteration_count=6000, beta=0.9)

nn = NeuralNetWithRMSprop(train_x, train_y, hidden_layers=(2, 2), iteration_count=6000, beta=0.9, epsilon=0.00000001)
```
### Hyper parameters

| parameter           | type  | sample value | description                              |     |
| ------------------- | ----- | ------------ | ---------------------------------------- | --- |
| hidden_layers       | tuple | (12, 24, 6)  | number of neurons in each layer          |     |
| learning_rate       | float | 0.01         | learning rate                            |     |
| iteration_count     | int   | 1000         | gradient descent iteration count         |     |
| learning_rate_decay | int   | 0.2          | learning rate decay value for each epoch |     |
| lambd               | float | 0.4          | regularization parameter                 |     |
| minibatch_size      | int   | 32           | mini batch size                          |     |
| shuffle             | bool  | True         | shuffle training data or not             |     |
| epochs              | int   | 5            | number of epochs                         |     |
| normalize_inputs    | bool  | True         | normalize inputs using zscore            |     |
|                     |       |              |                                          |     |

# Testing with multiple parameters and analysing results

One of the challenging jobs in machine learning is hyper parameter tuning. 

aibrite-ml provides NeuralNetAnalyser class to analyse prediction/train performance simultaneously. 

NeuralNetAnalyser uses different processors to train different models.

```python
import pandas as pd
import numpy as np

from aibrite.ml.neuralnet import NeuralNet
from aibrite.ml.neuralnetwithadam import NeuralNetWithAdam
from aibrite.ml.analyser import NeuralNetAnalyser

df = pd.read_csv("./data/winequality-red.csv", sep=";")

np.random.seed(5)
data = df.values

train_set, test_set, dev_set = NeuralNet.split(
    data, 0.6, 0.20, 0.20, shuffle=True)

train_x, train_y = (train_set[:, 0:-1]), train_set[:, -1]
dev_x, dev_y = (dev_set[:, 0:-1]), dev_set[:, -1]
test_x, test_y = (test_set[:, 0:-1]), test_set[:, -1]

# model configurations
normalize_inputs = [True, False]
iteration_count = [50, 100, 150]
learning_rate = [0.005, 0.002]
hidden_layers = [(32, 64, 128), (4, 4)]
lambds = [0.4, 0.8, 0.9]

# different test sets
test_sets = {'dev': (dev_x, dev_y),
             'test': (test_x, test_y),
             'train': (train_x, train_y)}


analyser = NeuralNetAnalyser("Red Wine Analysis")


for it in iteration_count:
    for lr in learning_rate:
        for hl in hidden_layers:
            for lambd in lambds:
                for ni in normalize_inputs:
                    analyser.submit(NeuralNetWithAdam, (train_x, train_y), test_sets,
                                    hidden_layers=hl,
                                    learning_rate=lr,
                                    iteration_count=it,
                                    lambd=lambd,
                                    normalize_inputs=ni)

analyser.join()
analyser.print_summary()
```

NeuralNetAnalyser.print_summary() prints model performance and recommendations.

```none
Waiting for 73 models to run ...

Model 49: *BEST* on [test, dev, __overall__]:0.65, 0.99, 0.88
--------------------------------------------------------------------------------
learning_rate       :0.005              hidden_layers       :(32, 64, 128)
iteration_count     :150                learning_rate_decay :0
lambd               :0.4                minibatch_size      :0
shuffle             :False              epochs              :1
normalize_inputs    :True               beta1               :0.9
beta2               :0.999              epsilon             :1e-08


                    ........................................
                             Prediction performance
                    ........................................
                        f1      prec    recall   support    time   change%
dev                   0.99      0.99      0.99       320    0.00     0.00
test                  0.65      0.65      0.66       320    0.00     0.00
train                 1.00      1.00      1.00       959    0.01   -0.00↓


                    ........................................
                               Train performance
                    ........................................
cost                  2.42
train time            4.80


                    ........................................
                            hyper parameter tunings
                    ........................................
                   current       new      dev f1     test f1    train f1
------------------------------------------------------------------------
iteration_count        150        50     -22.05↓      -9.65↓     -21.62↓     -18.81↓
------------------------------------------------------------------------------------
iteration_count        150       100      -2.55↓      -1.61↓      -2.63↓      -2.34↓
------------------------------------------------------------------------------------
normalize_inputs         1         0     -41.69↓     -15.03↓     -42.79↓     -35.49↓
------------------------------------------------------------------------------------
lambd                  0.4       0.8      -0.64↓      -1.59↓      -0.21↓      -0.71↓
------------------------------------------------------------------------------------
lambd                  0.4       0.9      -3.18↓      -1.53↓      -0.31↓      -1.69↓
------------------------------------------------------------------------------------
hidden_layers   (32, 64, 128)    (4, 4)     -41.32↓     -15.85↓     -39.53↓     -34.33↓
---------------------------------------------------------------------------------------
learning_rate        0.005     0.002      -1.90↓      -7.79↓       0.00↑      -2.64↓
--------------------------------------------------------------------------------

```

Above results shows the importance of input normalization and network size on model performance.