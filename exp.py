
from hubnet import *
from networkManager import *

NM = NetworkManager(filepath = 'iris_params.csv')
params = NM._readCSV()

n = Network([4, 5, 5, 3], params)

# n.train(d.X, d.Y, stopLoss=10, _print = False)

# Slices to try:

'''
n.graph(granularity = 100, sclices = [9, 40], sclice_positions=['x', 0, 'y', 0], colorMode = 'blend')
n.graph(granularity = 100, sclices = [9, 'x', 'y', 40], colorMode = 'blend')
n.graph(granularity = 100, sclices = ['y', 0, 7, 'x'], colorMode = 'blend')
n.graph(granularity = 100, sclices = ['y', 25, 34, 'x'], colorMode = 'blend', threshold = 190)




'''


def randomSlice(inputNum):
    
    xIndex = random.randint(0, inputNum-1)
    yIndex2 = random.randint(0, inputNum-1)

    while yIndex2 == xIndex:
        yIndex2 = random.randint(0, inputNum-1)

    l = [0] * inputNum
    l[xIndex] = 'x'
    l[yIndex2] = 'y'

    for i in range(inputNum):
        if l[i] == 0:
            l[i] = random.randint(0, 49)

    print("Generated Slice:", l)
    return l




# n.graph(granularity = 100, sclices = randomSlice(4), colorMode = 'blend', threshold = 150)
# n.graph(granularity = 100, sclices = [18, 42, 'x', 'y'], colorMode = 'blend', threshold = 220)
n.graph(granularity = 75, sclices = ['x', 'y', 34, 48], colorMode = 'blend')


