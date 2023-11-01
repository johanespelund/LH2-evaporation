import numpy as np

def smooth_transition(x, width=1.0):
    """
    Smooth transition function using a sigmoid.
    
    :param x: Spatial coordinate (can be a numpy array or a scalar)
    :param width: Width of the transition region
    :return: A value that goes smoothly from 0 to 1
    """
    return 1 / (1 + np.exp(-x / width))

def density(x, T, p, R, rho_liquid, transition_width):
    """
    Compute the density profile for a 1D phase change model with a smooth transition.
    
    :param x: Spatial coordinate (can be a numpy array or a scalar)
    :param T_gas: Temperature in the gas phase (assumed constant)
    :param p_gas: Pressure in the gas phase (assumed constant)
    :param rho_liquid: Density in the liquid phase (assumed constant)
    :param transition_width: Width of the transition region between phases
    :return: Density profile
    """
    # Ideal gas density for x >= 0
    rho_gas = p / (R * T)
    
    # Smooth transition between phases
    transition = smooth_transition(x, width=transition_width)
    
    # Density profile using numpy arrays
    rho = rho_liquid * (1 - transition) + rho_gas * transition

    return rho

