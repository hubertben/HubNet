
from hubnet import *
import random


class Dataset:

    def __init__(self, path, trainingPercent):
        self.path = path
        self.trainingPercent = trainingPercent

        self.data = []
        self.X = []
        self.Y = []

        self._openCSV()
        self._formatCSV_Iris()
        self.split()

    def _openCSV(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip().split(',') for line in lines]
        lines.pop(0)
        self.data = lines

    def _formatCSV_Iris(self):
        for line in self.data:
            line.pop(0) 
            line[0] = float(line[0])
            line[1] = float(line[1])
            line[2] = float(line[2])
            line[3] = float(line[3])

    def shuffle(self):
        random.shuffle(self.data)

    def seperateXY(self):
        self.X = [line[:-1] for line in self.data]
        self.Y = [line[-1] for line in self.data]

    def normalize(self):
        for j in range(len(self.X[0])):
            _max = 0
            _min = 0
            for i in range(len(self.X)):
                if self.X[i][j] > _max:
                    _max = self.X[i][j]
                if self.X[i][j] < _min:
                    _min = self.X[i][j]
            for i in range(len(self.X)):
                self.X[i][j] = (self.X[i][j] - _min) / (_max - _min)

        G = []
        for y in self.Y:
            if y == 'Iris-setosa':
                G.append([1, 0, 0])
            elif y == 'Iris-versicolor':
                G.append([0, 1, 0])
            else:
                G.append([0, 0, 1])
            
        self.Y = G

    def split(self):
        self.shuffle()
        self.seperateXY()
        self.normalize()
        self.X = self.X[:int(len(self.X) * self.trainingPercent)]
        self.Y = self.Y[:int(len(self.Y) * self.trainingPercent)]



d = Dataset('Iris.csv', .8)

n = Network(4, [5, 5, 3])

n.train(d.X, d.Y, stopLoss=50, _print = True)

n.graph(granularity=30, grab=2) # grab = 3 does not work

# def map(x, in_min, in_max, out_min, out_max):
#     return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# granularity = 15
# def generatePoints(cur = [], n = 0):

#     if(n == 0):
#         return cur

#     points = []

#     for i in range(granularity):
#         g = generatePoints(cur + [i], n - 1)
#         points.extend(g)

#     return points

# grab = 3
# poi = generatePoints([], grab)
# print(len(poi))

# points = []

# for p in range(0, len(poi), grab):
    
#     l = []
#     for i in range(grab):
#         l.append(map(poi[p + i], 0, granularity, -1, 1))

#     points.append(l)
    
        
# print(points)

# for p in points:

#     color = self(p)

#     typ = True
    
#     if(type(color) == list):
#         maxIndex = color.index(max(color))
#         color = colors[maxIndex]
#         typ = False
#         drawPoint(x, y, color)

#     else:

#         color = color.value
#         b = map(color, -1, 1, 0, 1)   

#         def RGBtoHEX(r, g, b):
#             r = int(map(r, 0, 1, 0, 255))
#             g = int(map(g, 0, 1, 0, 255))
#             b = int(map(b, 0, 1, 0, 255))
#             return '#%02x%02x%02x' % (r, g, b)

#         minB = min(points, key=lambda x: x[2])[2]
#         maxB = max(points, key=lambda x: x[2])[2]

#         for x, y, b in points:
#             b = map(b, minB, maxB, 0, 1)
#             color = RGBtoHEX(0, b, b)
#             drawPoint(x, y, color)       



             