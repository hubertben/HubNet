
import math
import random
import matplotlib.pyplot as plt
import os

from tkinter import *

colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'black']

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class Tensor:

    def __init__(self, value, children = (), operation = ''):
        self.value = value if isinstance(value, (int, float)) else value.value
        self.children = children
        self.operation = operation

        self.previous = None
        self._backward = lambda: None

        self.gradient = 0

    def __repr__(self):
        return 'Tensor(%s)' % self.value

    def __str__(self):
        return 'Tensor(%s)' % self.value

    def __add__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        newTensor = Tensor(self.value + other.value, [self, other], '+')
        
        def _backward():
            self.gradient += newTensor.gradient
            other.gradient += newTensor.gradient
        
        newTensor._backward = _backward
        
        return newTensor

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
        

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        newTensor = Tensor(self.value * other.value, [self, other], '*')
        
        def _backward():
            self.gradient += other.value * newTensor.gradient
            other.gradient += self.value * newTensor.gradient
        
        newTensor._backward = _backward
        
        return newTensor

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        other = other if isinstance(other, Tensor) else Tensor(other)
        newTensor = Tensor(self.value ** other.value, [self], '**')
        
        def _backward():
            self.gradient += newTensor.gradient * (self.value ** (other.value - 1)) * other.value
        
        newTensor._backward = _backward
        
        return newTensor

    def __rpow__(self, other):
        return self.__pow__(other)

    def __rtruediv__(self, other):
        return other * self**-1

    def __rpow__(self, other):
        return other ** self

    def __gt__(self, other):
        return self.value > other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value



    def _relu(self):
        newTensor = Tensor(0 if self.value < 0 else self.value, [self], 'relu')
        
        def _backward():
            self.gradient += (newTensor.value > 0) * newTensor.gradient
        
        newTensor._backward = _backward
        
        return newTensor

    def _tanh(self):
        n = (math.exp(self.value * 2) - 1)
        d = (math.exp(self.value * 2) + 1)
        t = n / d
        newTensor = Tensor(t, [self], 'tanh')
        
        def _backward():
            self.gradient += (1 - t**2) * newTensor.gradient
        
        newTensor._backward = _backward
        
        return newTensor


    def backward(self):
        topoOrder = []
        visited = set()

        def topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    topo(child)
                topoOrder.append(v)
        topo(self)

        self.gradient = 1
        for tensor in reversed(topoOrder):
            tensor._backward()

    def __round__(self, n):
        return round(self.value, n)

class Neuron:

    def __init__(self, inputCount):
        self.inputCount = inputCount
        self.weights = [Tensor(random.uniform(-1, 1)) for _ in range(inputCount)]
        self.bias = Tensor(0)

    def __call__(self, inputs):
        assert len(inputs) == self.inputCount
        return self.forward(inputs)

    def forward(self, inputs):
        activations = sum((weight * _input for weight, _input in zip(self.weights, inputs)), self.bias)
        return activations._tanh()

    def _parameters(self):
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return 'Neuron(%s)' % self.inputCount

    def _printWeights(self):
        for i, weight in enumerate(self.weights):
            print(' ------ Weight', i, ':', round(weight, 5), '\t| Grad:', round(weight.gradient, 5))

    def _printBias(self):
        print(' ---- Bias: %s' % self.bias)

    def _printParameters(self):
        print(' ---- Weights:')
        self._printWeights()
        self._printBias()


class Layer:

    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount

        self.neurons = [Neuron(inputCount) for _ in range(outputCount)]

    def __call__(self, inputs):
        output = []
        for neuron in self.neurons:
            output.append(neuron(inputs))
        return output[0] if len(output) == 1 else output

    def _parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron._parameters()]

    def __repr__(self) -> str:
        return 'Layer(%d, %d)' % (self.inputCount, self.outputCount)

    def _printParameters(self):
        for i, neuron in enumerate(self.neurons):
            print(' -- Neuron %d:' % i)
            neuron._printParameters()
        
class Network:

    def __init__(self, networkLayerSizes, parameters=None):
        self.totalNetwork = networkLayerSizes
        self.layers = []
        for layer in range(len(networkLayerSizes) - 1):
            self.layers.append(Layer(self.totalNetwork[layer], self.totalNetwork[layer + 1]))

        self.lossLog = []
        self.totalGraphedPoints = []

        if(type(parameters) == list):
            self.setParameters(parameters)
        elif(type(parameters) == Network):
            self.copyParameters(parameters)

    def __call__(self, inputs):
        output = 0
        for layer in self.layers:
            inputs = layer(inputs)
        output = inputs
        return output

    def _parameters(self):
        return [parameter for layer in self.layers for parameter in layer._parameters()]

    def setParameters(self, parameters):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                neuron.weights = parameters[:len(neuron.weights)]
                parameters = parameters[len(neuron.weights):]
                neuron.bias = parameters[0]
                parameters = parameters[1:]

    def copyParameters(self, other):
        self.setParameters(other._parameters())

    def _zeroGradients(self):
        for p in self._parameters():
            p.gradient = 0

    def __repr__(self) -> str:
        return 'Network(%s)' % self.layers

    def _printParameters(self):
        for i, layer in enumerate(self.layers):
            print('-Layer %d:' % i)
            layer._printParameters()

    def loss(self, yTruth, yPredicted):
        
        if(type(yPredicted[0]) == Tensor):
            loss = sum((yp - yt) ** 2 for yt, yp in zip(yTruth, yPredicted))
        else:
            loss = 0
            for yt, yp in zip(yTruth, yPredicted):
                _loss = sum((yp_ - yt_) ** 2 for yt_, yp_ in zip(yt, yp))
                loss += _loss

        self.lossLog.append(loss.value)
        return loss


    def updateWeights(self, learningRate):
        for p in self._parameters():
            p.value += -learningRate * p.gradient

    def trainInstance(self, xs, ys, learningRate):
        ypred = [self(x) for x in xs]
        loss = self.loss(ys, ypred)
        self._zeroGradients()
        loss.backward()
        self.updateWeights(learningRate)

        return loss


    def predict(self, x):
        ypred = self(x)
        return ypred

    
    def train(self, xs, ys, _print = None, stopLoss=0.01, learningRate=0.001, epochs=0):
        
        if(epochs == 0):

            topLoss = self.trainInstance(xs, ys, learningRate)
            lastP = 0

            while(True):
                loss = self.trainInstance(xs, ys, learningRate)
                
                if(_print == True):
                    print('Loss: %s' % loss.value)
                elif(_print == False):
                    l = round(100 - (map(loss.value, stopLoss, topLoss.value, 0, 100)), 2)
                    if(lastP != l):
                        print('Loss Percent: %s' % l)
                        lastP = l

                if loss.value < stopLoss:
                    break
        else:
            for i in range(epochs):
                loss = self.trainInstance(xs, ys, learningRate)
                
                if(_print):
                    print('Epoch %d: Loss: %s' % (i, loss.value))

                if loss.value < stopLoss:
                    break
                    
        return loss.value

            
    def _viewLoss(self):
        plt.plot(self.lossLog)
        plt.title('Loss')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.show()


    def drawPoint(self, canvas, x, y, r, color):
        canvas.create_rectangle(
            (x * self.block_size) + (r / 2), 
            (y * self.block_size) + (r / 2),
            ((x + 1) * self.block_size) - (r / 2), 
            ((y + 1) * self.block_size) - (r / 2), 
            fill=color, 
            outline=color,
            tags='%d, %d' % (x, y)
        )



    def graphMulti(self, graphNum, granularity, sclices = [], sclice_positions = []):
        
        self.graphNum = graphNum

        self.kint = Tk()
        self.kint.title('Network Graph')
        self.kint.resizable(False, False)
        self.geo = 1000
        self.kint.geometry('%dx%d' % (self.geo, self.geo))
        
        self.canvas_s = []

        s = math.sqrt(self.graphNum)

        for i in range(self.graphNum):
            canvas = Canvas(self.kint, width=int(self.geo // s), height=int(self.geo // s))
            canvas.grid(row=int(i // s), column=int(i % s))
            self.canvas_s.append(canvas)
        
        self.granularity = granularity
        self.block_size = (self.geo / self.granularity) // 10


        for graph in range(self.graphNum):

            print("Generating Graph %d" % graph)

            graph = map(graph, 0, self.graphNum, 0, self.granularity)
    
            points = []

            for x in range(granularity):
                for y in range(granularity):

                    l = []
                    
                    x_ = round(map(x, 0, granularity, -1, 1), 2)
                    y_ = round(map(y, 0, granularity, -1, 1), 2)

                    sclices_ = []
                    for s in sclices:
                        sclices_.append(round(map(s, 0, granularity, -1, 1), 2))
                    
                    c = 0
                    if(sclice_positions != []):
                        l = [0] * len(sclice_positions)

                        for i, s in enumerate(sclice_positions):
                            if(s == 'x'):
                                l[i] = x_
                            elif(s == 'y'):
                                l[i] = y_
                            elif(s == 'm'):
                                l[i] = map(graph, 0, self.graphNum, -1, 1)
                            else:
                                l[i] = sclices_[c]
                                c += 1
        
                    else:
                        l = [x_, y_]
                        l.extend(sclices_) 
                    
                    forward = self(l)

                    maxIndex = forward.index(max(forward))
                    color = colors[maxIndex]
                    points.append((x, y, 0, color))
                    
            self.totalGraphedPoints.append(points)


        for i in range(len(self.canvas_s)):
            print("Rendering Graph %d" % i)
            for p in self.totalGraphedPoints[i]:
                self.drawPoint(self.canvas_s[i], p[0], p[1], p[2], p[3])

        self.kint.mainloop()


    def graph(self, granularity, sclices = [], colorMode = "flat", threshold = 255):

        self.kint = Tk()
        self.kint.title('Network Graph')
        self.kint.resizable(False, False)
        self.geo = 1000
        self.kint.geometry('%dx%d' % (self.geo, self.geo))
        self.canvas = Canvas(self.kint, width=self.geo, height=self.geo)
        self.canvas.pack()

        def getter(event):   
            dir_name = "graphs/"
            term = os.listdir(dir_name)

            for item in term:
                if item.endswith(".ps"):
                    os.remove(os.path.join(dir_name, item))

            path = "graphs/" + str(sclices) + "_mode_" + colorMode + "_thr_" + str(threshold) + "_gran_" + str(granularity)
            from PIL import Image
            self.canvas.postscript(file=path + ".ps", colormode='color')
            img = Image.open(path + ".ps") 

            img.save(path + ".png", "png")

        self.canvas.bind("<Button-1>", getter)
        self.canvas.pack()

        self.granularity = granularity
        self.block_size = (self.geo / self.granularity)

        points = []

        print("Generating Points")

        for x in range(granularity):
            for y in range(granularity):

                l = []
                
                x_ = round(map(x, 0, granularity, -1, 1), 2)
                y_ = round(map(y, 0, granularity, -1, 1), 2)
                
                l = [0] * len(sclices)

                for i, s in enumerate(sclices):
                    if(s == 'x'):
                        l[i] = x_
                    elif(s == 'y'):
                        l[i] = y_
                    else:
                        l[i] = round(map(sclices[i], 0, granularity, -1, 1), 3)
                        
                forward = self(l)

                color = None

                if(colorMode == "flat"):
                    
                    maxIndex = forward.index(max(forward))
                    color = colors[maxIndex]

                elif(colorMode == "blend"):
                    
                    def RGBtoHEX(r, g, b):
                        return '#%02x%02x%02x' % (r, g, b)

                    r = map((forward[0].value), -1, 1, 0, 255)
                    g = map((forward[1].value), -1, 1, 0, 255)
                    b = map((forward[2].value), -1, 1, 0, 255)

                    if(threshold != 255):
                        if(r > threshold and g > threshold and b > threshold):
                            r = 255
                            g = 255
                            b = 255
                        elif(r > threshold and g > threshold):
                            r = 255
                            g = 255
                            b = 0
                        elif(r > threshold and b > threshold):
                            r = 255
                            g = 0
                            b = 255
                        elif(g > threshold and b > threshold):
                            r = 0
                            g = 255
                            b = 255
                        elif(r > threshold):
                            r = 255
                            g = 0
                            b = 0
                        elif(g > threshold):
                            r = 0
                            g = 255
                            b = 0
                        elif(b > threshold):
                            r = 0
                            g = 0
                            b = 255

                    color = RGBtoHEX(int(r), int(g), int(b))

                points.append((x, y, 0, color))
                
        self.totalGraphedPoints = points

        print("Rendering Points")

        for p in self.totalGraphedPoints:
            self.drawPoint(self.canvas, p[0], p[1], p[2], p[3])

        self.kint.mainloop()
        
