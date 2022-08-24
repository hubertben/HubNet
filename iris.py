
from hubnet import *
from networkManager import *

NM = NetworkManager(filepath = 'iris_params.csv')
params = NM._readCSV()

n = Network([4, 5, 5, 3], params)

# n.train(d.X, d.Y, stopLoss=10, _print = False)

n.graphMulti(
    graphNum = 50,
    granularity = 100, 
    sclices = [17],
    sclice_positions=['x', 'm', 'y', 'm']
)
