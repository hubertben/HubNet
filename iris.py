
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