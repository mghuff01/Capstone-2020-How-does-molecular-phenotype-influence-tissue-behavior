import numpy as np
import math
import pdb

class NeuronMB:

     def __init__(self, maxIter, O2max, prolif, apop, necro, rates):
        self.maxIter = maxIter
        self.O2max = O2max
        self.prolif = prolif
        self.apop = apop
        self.necro = necro
        self.rates = rates

        self.y1B, self.y1D = rates[0], rates[1]
        self.y20B, self.y20D = rates[2], rates[3]
        self.y1, self.y20 = self.y1B - self.y1D, self.y20B - self.y20D
        self.x1, self.x20 = 1, 20

     def load(self, NmatA, NmatD, NAtotal, NDtotal, eta, NH, Cs, n):
        self.NmatA = NmatA
        self.NmatD = NmatD
        self.NAtotal = NAtotal
        self.NDtotal = NDtotal
        self.eta = eta
        self.NH = NH
        self.Cs = Cs
        self.n = n


class Oligodentrocyte:

    def __init__(self, maxIter, O2max, prolif, apop, necro, rates):
        self.maxIter = maxIter
        self.O2max = O2max
        self.prolif = prolif
        self.apop = apop
        self.necro = necro
        self.rates = rates

        self.y1B, self.y1D = rates[0], rates[1]
        self.y20B, self.y20D = rates[2], rates[3]
        self.y1, self.y20 = self.y1B - self.y1D, self.y20B - self.y20D
        self.x1, self.x20 = 1, 20

    def load(self, NmatA, NmatD, NAtotal, NDtotal, eta, NH, Cs, n):
        self.NmatA = NmatA
        self.NmatD = NmatD
        self.NAtotal = NAtotal
        self.NDtotal = NDtotal
        self.eta = eta
        self.NH = NH
        self.Cs = Cs
        self.n = n
        
class ShedObj:

    def __init__(self, marker, NmatA, NmatD, n, maxIter):
        self.marker = marker
        self.NmatA = NmatA
        self.NmatD = NmatD
        if n != 1:
            self.NAtotal = np.sum(NmatA, 1)
            self.NDtotal = np.sum(NmatD, 1)
        else:
            self.NAtotal = NmatA
            self.NDtotal = NmatD
            
        self.n = n
        self.maxIter = maxIter
