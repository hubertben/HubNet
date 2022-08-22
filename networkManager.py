
from hubnet import *

class NetworkManager:

    def __init__(self, network = None, filepath = ""):

        self.network = network
        self.filepath = filepath

    def _writeCSV(self):
        with open(self.filepath, 'w') as f:
            for i in self.network._parameters():
                f.write(str(i.value) + '\n')

    def _readCSV(self):
        lines = []
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        stripped = []
        for l in lines:
            l = l.strip('\n')
            stripped.append(Tensor(float(l)))

        return stripped
            
        
            
        