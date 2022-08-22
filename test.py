
from hubnet import *

xs = [                      # input
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
]

ys = [-1, 1, 1, -1]         # output

n = Network(2, [3, 3, 1])   # 2 inputs, 2 - (3 Node) hidden layers, 1 output

# n._printParameters()      # prints all nodes of all layers of network, 
                            # along with their gradients

n.train(
    xs, 
    ys, 
    _print = True,          # True: prints loss, False: prints loss percentage
    stopLoss = .01,         # Stop when loss is less than this
    learningRate = 0.01,    # Learning rate
    epochs=0                # 0: use stoploss, >0: use this many epochs
)

p = n.predict([.25, .75])   # test network on specific input
print(p)                  

n.graph(
    granularity=50          # Granularity of graph
)

n._viewLoss()               # Shows loss of network

