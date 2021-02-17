#MS Comparmentalization Model Figures

#Library Imports
import pandas as pd
import numpy as np
import scipy as sci
import pdb
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import math

#Data
class GrowObj:

    def __init__(self, maxIter, kv, O2max, sigma, rates):
        self.maxIter = maxIter
        self.kv = kv
        self.O2max = O2max
        self.sigma = sigma
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


class MeshObj:

    def __init__(self, maxIter, kvs, O2maxs, sigma, rates):
        self.maxIter = maxIter
        self.kvs = kvs
        self.O2maxs = O2maxs
        self.sigma = sigma
        self.rates = rates

        self.y1B, self.y1D = rates[0], rates[1]
        self.y20B, self.y20D = rates[2], rates[3]
        self.y1, self.y20 = self.y1B - self.y1D, self.y20B - self.y20D
        self.x1, self.x20 = 1, 20

    def load(self, grow_objs, idxValid, aEnd, nVec, paramSet, a_matValid, d_matValid, eta_matValid):
        self.grow_objs = grow_objs
        self.idxValid = idxValid
        self.aEnd = aEnd
        self.nVec = nVec
        self.paramSet = paramSet
        self.a_matValid = a_matValid
        self.d_matValid = d_matValid
        self.eta_matValid = eta_matValid
    

# Subsystem 1 - Tumor Growth
def ST_grow(grow_obj, dt=1):
    """
    Main model to run solid tumor (ST) growth simulations.
    """

    # Constants
    #-----------
    wiggle = 0.01 # numerical stability

    # size of system/local behavior around vasculature point (mm)
    # single cell thickness for vasc case = min granularity (10µm)
    sigma = grow_obj.sigma
    eps = 0.1  # in mm (0.1mm = 100µm), 0.01mm = 10µm(cell)

    # cellular density
    rho = 1 * (10 ** 6)
    p = 0.2  # 20% from Hori paper

    # Growth rate data
    y1B, y1D = grow_obj.y1B, grow_obj.y1D
    y20B, y20D = grow_obj.y20B, grow_obj.y20D
    y1, y20 = grow_obj.y1, grow_obj.y20
    x1, x20 = grow_obj.x1, grow_obj.x20

    # rate of angiogenesis(mm/cell, or amt recruited)
    ka = 10e-6  # can tune with all sorts of values

    # Relative vacular volume (RVV) data
    c = 0.17
    
    rmax = eps  # note: eventually replace ***

    # number of tumor compartments
    n = np.ceil(rmax / sigma).astype(numpy.int64)

    # number of plasma compartments
    # avg central vessel d = 57.5 µm in study...
    # changes over time, but know diameter gets bigger over time with tumor
    # 57.5 / 2 - 28.75. This translates to roughly "3 compartments" worth of space for the plasma/vasculature
    npl = 3

    # find physical capacity for n compartments 
    h = 5 * sigma # we arbitrarily set height to (eps/2)
    Cs, Ctot, fracs = computeCs(sigma, n, npl, h)

    # calculate model k parameters for all n instantaeous rates / parameter space

    # effective growth rates (gammas)
    # effective birth rates (betas)
    # effective death rates (deltas)
    kGs = compartmentRates(x1, x20, y1, y20, sigma, n, grow_obj.O2max)
    kBs = compartmentRates(x1, x20, y1B, y20B, sigma, n, grow_obj.O2max)
    kDs = compartmentRates(x1, x20, y1D, y20D, sigma, n, grow_obj.O2max)
    NH = calcHori(grow_obj.maxIter)

    # Data structure setup
    #----------------------
    # Growth data matrix setup - alive
    NmatA = np.zeros((grow_obj.maxIter, n))

    # Matrix preallocation(column vector for now)
    IC_1 = 1  # IC for comp 1
    IC_2TOn = 0  # IC for comps 2...n

    NmatA[0, 0] = IC_1  # populating with IC - comp 1

    # Necrosis data matrix setup / matrix preallocation
    NmatD = np.zeros((grow_obj.maxIter, n))

    # Monoexponential Growth
    #-----------------------
    # net growth function (alive/proliferative)
    NAi_t = lambda t, Ni, kGi:  Ni[t - 1] * (1 + kGi*dt)

    # necrosis function (death)
    NDi_t = lambda t, Ni, kDi:  Ni[t] * kDi*dt

    # functions to assert physical capacity of tumor space as an upper bound
    excessC = lambda Ncurr, C: int(Ncurr > C) * (Ncurr - C)
    assertC = lambda Ncurr, C: max(int(Ncurr <= C) * Ncurr, int(Ncurr > C) * C)

    # store overflow values (at time t) of each compartment i
    excessMat = np.zeros((grow_obj.maxIter, n))

    # using a boolean array to notify activity of compartment
    active = [False] * n
    active[0] = True

    # Run model - recursive difference equations
    #--------------------------------------------
    for s in range(2, grow_obj.maxIter + 1):
        for j in range(1, n + 1):
            t = s - 1
            i = j - 1  # dummy var to match MATLAb indexing

            # calculate growth at current time step
            Ni_curr = NAi_t(t, NmatA[:, i], kGs[i])

            # adding overflow / cell spillage of neighboring / previous comp(i - 1)
            if j > 1:
                Ni_curr = Ni_curr + excessMat[t, i - 1]

            # calculating overflow/spillover if reaching C of current comp (i)
            excessMat[t, i] = excessC(Ni_curr, Cs[i])

            if (j + 1 <= n) and (active[i + 1] == False) and (excessMat[t, i] > 0):
                active[i + 1] = True

            # assertions to hit physical capacity
            asserts = assertC(Ni_curr, Cs[i])

            # matrix update
            NmatA[t, i] = np.max([asserts, IC_2TOn])
            # lower bound asserted
            NmatD[t, i] = NDi_t(t, NmatA[:, i], kDs[i])
            
            # new: probabilistic movement
            prob_in = 0.10 # slightly more incentive due to oxygenation
            prob_out = 0.05
            prob_dump = 0.10
                
            if (t > 0) and (i < 9):
                
                # factor of 10 difference between compartments ==> dump
                if ((NmatA[t,i] / (NmatA[t, i+1] + wiggle)) > 10):
                    Ni_dump = prob_dump*Ni_curr

                    Ni_currp = Ni_curr - Ni_dump
                    Ni_plus1 = NmatA[t, i+1] + Ni_dump

                    # update
                    NmatA[t, i+1], Ni_curr = Ni_plus1, Ni_currp
                
                Ni_out = prob_out*Ni_curr
                Ni_in = prob_in*NmatA[t, i+1] 
                
                if Ni_curr > 10:
                    indicator_out = 1
                else:
                    indicator_out = 0
                
                if NmatA[t, i+1] > 10:
                    indicator_in = 1
                else:
                    indicator_in = 0
                    
                Ni_currp = Ni_curr + indicator_in*Ni_in - indicator_out*Ni_out
                Ni_plus1 = NmatA[t, i+1] - indicator_in*Ni_in + indicator_out*Ni_out
                
                # update
                NmatA[t, i+1], Ni_curr = Ni_plus1, Ni_currp
            
            
            # using user-defined parameter
            h = grow_obj.kv * t
            # linear <-- last used: h = height(t)

            Cs, Ctot, fracs = computeCs(sigma, n, npl, h)
            # note: 0.01mm = 10µm(cell)

        # print(t, i, n)
        # print(active)

    # total alive population/unit time
    NAtotal = np.sum(NmatA, 1)

    # total dead population/unit time
    NDtotal = np.sum(NmatD, 1)

    # necrotic fraction/unit time (should increase over time)
    eta = np.divide(np.cumsum(NDtotal), (NAtotal + np.cumsum(NDtotal)))
    # double check: should we cumsum here?
    grow_obj.load(NmatA, NmatD, NAtotal, NDtotal, eta, NH, Cs, n)

    return grow_obj


def ST_grow_mesh(mesh_obj, dt=1):
    """
    Run Tumor Growth - over chosen parameter space mesh. Calls function above.
    """

    N = len(mesh_obj.O2maxs)
    M = len(mesh_obj.kvs)
    O2max_arr = np.reshape(
        np.repeat(np.array(mesh_obj.O2maxs).T, M), (M * N, 1))
    kv_arr = np.reshape(np.matlib.repmat(
        np.array(mesh_obj.kvs).T, N, 1).flatten(), (M * N, 1))

    if N == 1:
        paramSet = np.concatenate([O2max_arr, kv_arr], 1)
    else:
        paramSet = np.concatenate([O2max_arr.T, kv_arr.T], 0).T

    # index for output
    grow_objs = []
    for i in range(0, N):
        for j in range(0, M):
            grow_obj_i = GrowObj(mesh_obj.maxIter, mesh_obj.kvs[
                j], mesh_obj.O2maxs[i], mesh_obj.sigma, mesh_obj.rates)
            grow_objs.append(ST_grow(grow_obj_i, dt=dt))

        print('\nFinished analyzing O2max = %s!\n' % (mesh_obj.O2maxs[i]))
    # check: k = N * M

    # Filter growth trajectories ...
    #--------------------------------
    # .. for valid parameterSet / runs that has specified nonzero compartments and overall population
    # n = gr(1).data.n

    print("\nFiltering out runs without sufficient growth and necrosis...\n")
    aValid = {}  # change to P
    dValid = {}  # change to N
    etaValid = {}

    idxValid = []
    aEnd = []
    nVec = []

    for l in range(0, len(grow_objs)):
        NA_final = grow_objs[l].NAtotal[-1]
        n_thresh = np.sum(np.any(grow_objs[l].NmatA, 0))
        aEnd.append(NA_final)
        nVec.append(n_thresh)

        # applying filters; to have all() instead of sum()?
        if n_thresh == 10 and NA_final >= 1e7:
            # pdb.set_trace()
            print("passing criteria")
            key = str(paramSet[l, :])
            aValid[key] = grow_objs[l].NAtotal
            dValid[key] = grow_objs[l].NDtotal
            etaValid[key] = grow_objs[l].eta

            print('Parameter coordinate pair %s (%s, %s) is valid!' % (
                l, paramSet[l, 0], paramSet[l, 1]))
            idxValid.append(l)

    print('=' * 30)

    # Convert run Map/dict to plotting matrix
    try:
        keysValid = paramSet[idxValid, :]
    except IndexError:
        print("# valid runs:", len(idxValid))
        print("\nNo valid runs found! Aborting... Try to increase lower limit of C_0")
        return

    a_matValid = []
    d_matValid = []
    eta_matValid = []

    for i in range(0, len(keysValid)):
        charValid = str(keysValid[i, :])
        a_matValid.append(aValid[charValid])
        d_matValid.append(dValid[charValid])
        eta_matValid.append(etaValid[charValid])

    mesh_obj.load(grow_objs, idxValid, aEnd, nVec, paramSet,
                  a_matValid, d_matValid, eta_matValid)

    return mesh_obj

    
