from ms_data import *

# import pandas as pd
import numpy as np
import numpy.matlib
import scipy as sci
import pdb

# Subsystem 1 - Neuron Mass Balance (NMB)
def N_shrink(shrink_NMB, dt=1):

    # Constants
    #----------

    #Growth Rate Data
    y1B, y1D = shrink_NMB.y1B, shrink_NMB.y1D
    y20B, y20D = shrink_NMB.y20B, shrink_NMB.y20D
    y1, y20 = shrink_NMB.y1, shrink_NMB.y20
    x1, x20 = shrink_NMB.x1, shrink_NMB.x20

    #Compartments
    n = 7

    #Rate Constants: Differentiate Later into sub-compartments
    #N = Neurons
    #A - Apoptosis
    #N - Necrosis
   
    #Neurons: Assumed not to proliferate, kPN = 0
    kAN = compartmentRates(x1, x20, y1D, y20D, n, shrink_NMB.O2max)
    kNN = compartmentRates(x1, x20, y1D, y20D, n, shrink_NMB.O2max)

    #Placeholder Functions for Apoptosis and Necrosis in Oligodendrocyte/Neuron
    fAN=0
    fNN=0

    # Data structure setup
    #----------------------
    # Growth data matrix setup - alive
    NmatA = np.zeros((shrink_NMB.maxIter, n))

    # Matrix preallocation(column vector for now)
    IC_1 = 1  # IC for comp 1
    IC_2TOn = 0  # IC for comps 2...n

    NmatA[0, 0] = IC_1  # populating with IC - comp 1

    # Necrosis data matrix setup / matrix preallocation
    NmatD = np.zeros((shrink_NMB.maxIter, n))

    # Neuron Mass Balance Equation
    #----------------------------------------
    # Neuron (death = Apoptosis and Necrosis) 
    NDi_t = lambda t, Ni, kGi: fAN + fNN

    ############### using a boolean array to notify activity of compartment
    active = [False] * n
    active[0] = True

    # Run model 
    #--------------------------------------------
    for s in range(1, shrink_NMB.maxIter + 1):
        for j in range(1, n + 1):
            t = s - 1
            i = j - 1  # dummy var to match MATLAb indexing

            # matrix update
            NmatA[t, i] = np.max([asserts, IC_1])
            # lower bound asserted
            NmatD[t, i] = NDi_t(t, NmatA[:, i], kDs[i])

            # using user-defined parameter
            h = shrink_NMB.kv * t
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
    shrink_NMB.load(NmatA, NmatD, NAtotal, NDtotal, eta, NH, Cs, n)

    return shrink_NMB

# Subsystem 2 - Oligodendrocyte Mass Balance (LMB)
def L_shrink(shrink_LMB, dt=1):

    # Constants
    #----------

    #Growth Rate Data
    y1B, y1D = shrink_LMB.y1B, shrink_LMB.y1D
    y20B, y20D = shrink_LMB.y20B, shrink_LMB.y20D
    y1, y20 = shrink_LMB.y1, shrink_LMB.y20
    x1, x20 = shrink_LMB.x1, shrink_LMB.x20

    #Compartments
    n = 7

    #Rate Constants: Differentiate Later into sub-compartments
    #L = Oligodendrocyte
    #P - Proliferation
    #A - Apoptosis
    #N - Necrosis
   
    #Oligodendrocytes
    kPL = compartmentRates(x1, x20, y1B, y20B, n, shrink_LMB.O2max)
    kAL = compartmentRates(x1, x20, y1D, y20D, n, shrink_LMB.O2max)
    kNL = compartmentRates(x1, x20, y1D, y20D, n, shrink_LMB.O2max)

    #Placeholder Functions for Apoptosis and Necrosis in Oligodendrocyte/Neuron
    fPL=0
    fAL=0
    fNL=0

    # Data structure setup
    #----------------------
    # Growth data matrix setup - alive
    NmatA = np.zeros((shrink_LMB.maxIter, n))

    # Matrix preallocation(column vector for now)
    IC_1 = 1  # IC for comp 1
    IC_2TOn = 0  # IC for comps 2...n

    NmatA[0, 0] = IC_1  # populating with IC - comp 1

    # Necrosis data matrix setup / matrix preallocation
    NmatD = np.zeros((shrink_LMB.maxIter, n))

    # Oligodendrocyte Mass Balance Equation
    #----------------------------------------
    # Oligodendrocyte (birth)
    NAi_t = lambda t, Ni, kGi:  fPL

    # Oligodendrocyte (death = Apoptosis and Necrosis) 
    NDi_t = lambda t, Ni, kGi: fAL + fNL

    ############### using a boolean array to notify activity of compartment
    active = [False] * n
    active[0] = True

    # Run model 
    #--------------------------------------------
    for s in range(1, shrink_LMB.maxIter + 1):
        for j in range(1, n + 1):
            t = s - 1
            i = j - 1  # dummy var to match MATLAb indexing

            # matrix update
            NmatA[t, i] = np.max([asserts, IC_1])
            # lower bound asserted
            NmatD[t, i] = NDi_t(t, NmatD[:, i], kDs[i])

            # using user-defined parameter
            h = shrink_NMB.kv * t
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
    shrink_LMB.load(NmatA, NmatD, NAtotal, NDtotal, eta, NH, Cs, n)

    return shrink_LMB








    

def compartmentRates(x1, x20, y1, y20, sigma, n, O2max):
    """
    Assign rates of birth, death, and net growth rates for each compartment.
    """
    # first get the slope and intercept of the 2 data points, 1 % and 20%
    xvec = [x1, x20]
    # oxygen concs.
    yvec = [y1, y20]

    # G / B / D rates
    p = np.polyfit(xvec, yvec, deg=1)
    m = p[0]
    b = p[1]
    y = lambda x: np.multiply(m, x) + b
    # we assume linear mapping from [O2] to kG(while d to[O2] is nonlinear)

    # Max[O2]:12.00%, Min[O2]:0.255%
    C0 = O2max
    dhalf = 0.018
    Cd = lambda d: np.multiply(C0, np.power(0.5, np.divide(d, dhalf)))

    i_vals = list(range(1, n + 1))
    distData = np.multiply(np.divide((np.multiply(2, i_vals) - 1), 2))
    O2Data = Cd(distData)
    kGData = y(O2Data)

    return kGData

