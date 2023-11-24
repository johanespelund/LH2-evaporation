import numpy as np
from thermo import R

def sigma_evaporation(p_sat, T_l, T_g, p_g, DOF):
    """
    Calculate the evaporation coefficient sigma_e*.

    Parameters:
    p_sat (float): Saturation pressure at the liquid temperature (Pa)
    T_l (float): Liquid temperature (K)
    T_g (float): Gas temperature (K)
    p_g (float): Gas pressure (Pa)
    DOF (int): Degrees of freedom for the molecule

    Returns:
    float: Evaporation coefficient sigma_e*
    """
    return ((p_sat / p_g) *
            np.exp((DOF + 4) * (1 - (T_g / T_l))) *
            ((T_g / T_l) ** (DOF + 4)))

def sigma_condensation(T_l, T_g, DOF):
    """
    Calculate the condensation coefficient sigma_c*.

    Parameters:
    T_l (float): Liquid temperature (K)
    T_g (float): Gas temperature (K)
    DOF (int): Degrees of freedom for the molecule

    Returns:
    float: Condensation coefficient sigma_c*
    """
    return (np.sqrt(T_g / T_l) *
            np.exp(-(DOF + 4) * (1 - (T_g / T_l))) *
            ((T_g / T_l) ** (DOF + 4)))


def mass_flux_HKS(sigma_c, T_l, T_g, p_sat, p_g, M):
    """
    Calculate the mass flux using the Hertz-Knudsen-Schrage equation.

    Parameters:
    sigma_c (float): Condensation coefficient
    T_l (float): Liquid temperature (K)
    T_g (float): Gas temperature (K)
    p_sat (float): Saturation pressure at the liquid temperature (Pa)
    p_g (float): Gas pressure (Pa)

    Returns:
    float: Mass flux (kg/(m^2·s))
    """
    return (
        ((2 * sigma_c) / (2 - sigma_c)) *
        np.sqrt(M / (2 * np.pi * R)) *
        (p_sat / np.sqrt(T_l) - p_g / np.sqrt(T_g))
        )
    

def R_qq(C_eq, Tl, R, M):
    v_mp = np.sqrt((2*R*Tl)/M)     # [m/s]
    return ((np.sqrt(np.pi))/(4 * C_eq* R * Tl**2 * v_mp))*(1 + 104/(25*np.pi))

def R_qmu(C_eq, Tl, R, M):
    v_mp = np.sqrt((2*R*Tl)/M)     # [m/s]
    return ((np.sqrt(np.pi))/(8 * (C_eq) * Tl * v_mp))*(1 + 16/(5*np.pi))

def R_mumu(C_eq, Tl, R, M, Tg, DOF):
    v_mp = np.sqrt((2*R*Tl)/M)     # [m/s]
    sigma = sigma_condensation(Tl, Tg, DOF)
    return ((2 * (R)) * np.sqrt(np.pi))/((C_eq) * v_mp) * ((sigma)**(-1) + np.pi**(-1) - 23/32)    
   
    
if __name__ == "__main__":
    from thermo import M_water, DOF_water

    # Example usage for water at following state
    temp_l = 273.15 + 5.2  # Liquid temperature (K)
    temp_g = 273.15 + 5.6  # Gas temperature (K)
    p_sat = 883.8  # Saturation pressure (Pa)
    p_g = 882  # Gas pressure (Pa)
    DOF = DOF_water
    M = M_water

    # Calculate sigma_evaporation
    sigma_e = sigma_evaporation(p_sat, temp_l, temp_g, p_g, DOF)

    # Calculate sigma_condensation
    sigma_c = sigma_condensation(temp_l, temp_g, DOF)
    
    j_hks = mass_flux_HKS(sigma_c, temp_l, temp_g, p_sat, p_g, M)

    print(f"Sigma Evaporation: {sigma_e: .4f}")
    print(f"Sigma Condensation: {sigma_c: .4f}") 
    print(f"Mass flux, J_HKS: {j_hks: .2e} kg/(m^2*s)")
