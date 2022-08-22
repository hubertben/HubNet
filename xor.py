
from hubnet import *
from networkManager import *
import time

xs = [
    [-1, -1],
    [-1, 1],
    [1, -1],
    [1, 1],
]

ys = [-1, 1, 1, -1]
p = [.8, -.7]

arch = [4, 4, 1]
input_count = 2

n = Network(input_count, arch) # 2 inputs, 4 hidden, 1 output

NM = NetworkManager(network = n, filepath = 'xor.csv')

# With training the Network

start = time.time()

n.train(xs, ys, stopLoss=.01, _print = False)
print(n.predict(p))

end = time.time()
print('Ellapsed time (Train Weights):', end - start)


NM._writeCSV()
params = NM._readCSV()


# Without retraining the Network

start = time.time()

g = Network(input_count, arch, params)
print(g.predict(p))

end = time.time()
print('Ellapsed time (Read in Weights):', end - start)
