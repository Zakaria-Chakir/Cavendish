import numpy as np
import scipy 
import matplotlib.pyplot as plt


def Torsional_oscillator_model(t, C, damp, I, m_big, m_small, dist_mass, kappa, G, d, phi):
    b = 2*damp/I 
    kappa_eff = kappa + 2*G*d*((m_small*m_big)/(dist_mass))
    w0 = np.sqrt(kappa_eff/I)
    w1 = np.sqrt(w0^2-b^2)
    theta = C*np.e^(-b*t)* np.cos(w1*t+phi)
    return theta 

