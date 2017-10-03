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

# Testing multiple parameters and analysing reslts 