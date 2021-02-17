import numpy as np
from ms_model2 import *
import pdb
import pandas as pd
import pickle
from ms_plotter import *

#----------------
# Main functions
#----------------

def analyze_markerEC(shed_obj_Fe, shed_obj_Fne):

    maxIter = shed_obj_Fe.maxIter
  

    N = deserialize("N.obj")
    P = deserialize("P.obj")
    eta = deserialize("eta.obj")

    Ntot = np.reshape(np.sum(N, 1), (maxIter,))
    Ptot = np.reshape(np.sum(P, 1), (maxIter,))

    qp_e = np.reshape(shed_obj_Fe.qp, (maxIter, 1))
    qp_ne = np.reshape(shed_obj_Fne.qp, (maxIter, 1))
    
    qt_e = np.reshape(shed_obj_Fe.qt, (maxIter, 1))
    qt_ne = np.reshape(shed_obj_Fne.qt, (maxIter, 1))
    qtmat_e = shed_obj_Fe.qtmat
    qtmat_ne = shed_obj_Fne.qtmat
    
    n = qtmat_e.shape[1]

    t = np.divide(np.array(list(range(0, maxIter))), 365)
    
    # plot compartment contributions
    ylab = r'Protein mass in plasma (log$_{10}$ U)'
    xlab1 = r'Proliferative population (log$_{10}$ cells)'
    xlab2 = r'Necrotic population (log$_{10}$ cells)'
    titleStr = 'Compartmental contributions to EC\nprotein outflux over population growth'
    plot_comp_bar(qtmat_e, qt_e, Ptot, n, titleStr, xlab1, ylab, PN_flag="P", shed_flag=True) # Ptot used to be eta
    titleStr = 'Compartmental contributions to Non-EC\nprotein outflux over population growth'
    plot_comp_bar(qtmat_ne, qt_ne, Ntot, n, titleStr, xlab2, ylab, PN_flag="N", shed_flag=True) # Ntot used to be eta



def analyze_compete(shed_objs):
    """
    non-EC beating EC proteins.
    """

    maxIter = shed_objs[0].maxIter
   

    t = np.reshape(
        np.divide(np.array(list(range(0, maxIter))), 365), (maxIter, 1))

    N = deserialize("N.obj")
    P = deserialize("P.obj")
    Ntot = np.reshape(np.sum(N, 1), (maxIter, 1))
    Ptot = np.reshape(np.sum(P, 1), (maxIter, 1))

    qps = []
    for so in shed_objs:
        qp = np.reshape(so.qp, (maxIter, 1))
        qps.append(qp)
    
    X1 = np.matlib.repmat(Ptot, 1, len(qps))  # over population
    X2 = np.matlib.repmat(t, 1, len(qps))  # over time
    Y = np.concatenate(qps, axis=1)
    ylab = 'Protein mass in plasma (log$_{10}$ U)'

    # Single param boosting
    titleStr = 'Single parameter boosting over time'
    xlab = 'Time elapsed (years)'
    plot_compete(X2, Y, t, Ptot, titleStr, xlab, ylab, log_flag="semilogy")
    
    titleStr = 'Single parameter boosting over population growth'
    xlab = 'Proliferative population (log$_{10}$ cells)'
    plot_compete(X1, Y, t, Ptot, titleStr, xlab, ylab, log_flag="loglog")



def analyze_compete_grid(shed_objs,experiment=None):

    maxIter = shed_objs[0].maxIter
  

    P = deserialize("P.obj")
    Ptot = np.reshape(np.sum(P, 1), (maxIter, 1))
    
    sheds = []
    for so in shed_objs:
        qp = np.reshape(so.qp, (maxIter, 1))
        sheds.append(qp)

    t = np.reshape(np.divide(np.array(list(range(0, maxIter))), 365), (maxIter, 1))
    Y = np.concatenate(sheds, axis=1)

    # find min and max trajectories - note: all monotonic increasing
    # want per u_H
    
    low_uh_idx = []
    for i in range(2, len(sheds)):
        if (i-2) % 3 == 0:
            low_uh_idx.append(i)
            
    low_uh_idx = np.array(low_uh_idx)
    med_uh_idx = low_uh_idx + 1
    high_uh_idx = low_uh_idx + 2
    
    Y_low = Y[:, low_uh_idx]
    Y_med = Y[:, med_uh_idx]
    Y_high = Y[:, high_uh_idx]
    
    max_col_low = np.argmax(np.max(Y_low, axis=0))
    min_col_low = np.argmin(np.max(Y_low, axis=0))
    max_col_med = np.argmax(np.max(Y_med, axis=0))
    min_col_med = np.argmin(np.max(Y_med, axis=0))
    max_col_high = np.argmax(np.max(Y_high, axis=0))
    min_col_high = np.argmin(np.max(Y_high, axis=0))
    
    Y_max_low = Y_low[:, max_col_low]
    Y_min_low = Y_low[:, min_col_low]
    Y_max_med = Y_med[:, max_col_med]
    Y_min_med = Y_med[:, min_col_med]
    Y_max_high = Y_high[:, max_col_high]
    Y_min_high = Y_high[:, min_col_high]

    Y_new = np.concatenate([np.expand_dims(Y[:,0], axis=1),
                            np.expand_dims(Y[:,1], axis=1), 
                            np.expand_dims(Y_min_low, axis=1), 
                            np.expand_dims(Y_max_low, axis=1),
                            np.expand_dims(Y_min_med, axis=1), 
                            np.expand_dims(Y_max_med, axis=1),
                            np.expand_dims(Y_min_high, axis=1), 
                            np.expand_dims(Y_max_high, axis=1)], 
                           axis=1)
    
    X1 = np.matlib.repmat(t, 1, 8) 
    X2 = np.matlib.repmat(Ptot, 1, 8)  # over population
    
    xlab1 = 'Time elapsed (year)'
    xlab2 = 'Proliferative population (log$_{10}$ cells)'
    ylab = 'Protein mass in plasma (log$_{10}$ U)'
    titleStr1 = 'Parameter scans over time'
    titleStr2 = 'Parameter scans over population growth'
    
    plot_compete_grid(X1, Y_new, t, titleStr1, xlab1, ylab, experiment, log_flag="semilogy")
    plot_compete_grid(X2, Y_new, t, titleStr2, xlab2, ylab, experiment, log_flag="loglog")


def analyze_shed_sensitivity(sens_arr, params, pnames, P):
       
    maxIter = sens_arr[0][0].maxIter
    t = np.divide(np.array(list(range(0, maxIter))), 365)
    N = len(t)
    
    try:
        Ptot = np.sum(P, 1)
    except: #AxisError
        Ptot = P
    
    ylab = r"Protein mass in plasma (log$_{10}$ U)"
    xlab = r'Proliferative population (log$_{10}$ cells)'
    strformats = ["%.3f", "%d", "%.3f", "%.2f", "%.2f"]

    for i, shed_objs in enumerate(sens_arr):  # list of list of grow_objs
        tot_lines = len(shed_objs)
        half_lines = tot_lines // 2
        X = np.matlib.repmat(Ptot, half_lines, 1).T # used to be Ptot
        
        # proliferative and necrotic
        titleStr = 'Protein shedding sensitivity\nto ' + pnames[i] + ' over population growth'

        qp_list = [np.reshape(so.qp, (N, 1)) for so in shed_objs]
        Ye = np.concatenate(qp_list[:half_lines], axis=1)
        Yne = np.concatenate(qp_list[half_lines:], axis=1)
            
        plot_sens_shed(X, Ye, Yne, t, titleStr, xlab, ylab, Ptot, pnames[i], params[i], strformats[i])

#------------------
# Helper functions
#------------------

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)
