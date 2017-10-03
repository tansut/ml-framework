# Aibrite Machine Learning Framework 

## Basic usage
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
