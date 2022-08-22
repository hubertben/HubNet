
from hubnet import *

xs = [
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
]

ys = [-1, 1, 1, -1]

n = Network(2, [4, 4, 1]) # 2 inputs, 4 hidden, 1 output

n.train(xs, ys, stopLoss=.05, _print = True)

n.graph(granularity=20)