
from hubnet import *
from networkManager import *

NM = NetworkManager(filepath = 'iris_params.csv')
params = NM._readCSV()

n = Network([4, 5, 5, 3], params)

# n.train(d.X, d.Y, stopLoss=10, _print = False)

n.initCanvas(granularity=50)
n.graph(
    granularity = 50, 
    sclices = [35, 25],
    sclice_positions=[0, 'x', 0, 'y']
)
