import math
R = 8.314  # Universal gas constant in J/(mol·K)

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
            math.exp((DOF + 4) * (1 - (T_g / T_l))) *
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
    return (math.sqrt(T_g / T_l) *
            math.exp(-(DOF + 4) * (1 - (T_g / T_l))) *
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
        math.sqrt(M / (2 * math.pi * R)) *
        (p_sat / math.sqrt(T_l) - p_g / math.sqrt(T_g))
        )
    
if __name__ == "__main__":

    from thermopack.cubic import cubic

    water = cubic('H2O', 'SRK')
    M = water.compmoleweight(1) * 1e-3  # [kg/mol]

    # Example usage for water at following state
    temp_l = 273.15 + 5.2  # Liquid temperature (K)
    temp_g = 273.15 + 5.6  # Gas temperature (K)
    p_sat = 883.8  # Saturation pressure (Pa)
    p_g = 882  # Gas pressure (Pa)
    DOF = 3

    # Calculate sigma_evaporation
    sigma_e = sigma_evaporation(p_sat, temp_l, temp_g, p_g, DOF)

    # Calculate sigma_condensation
    sigma_c = sigma_condensation(temp_l, temp_g, DOF)
    
    j_hks = mass_flux_HKS(sigma_c, temp_l, temp_g, p_sat, p_g, M)

    print(f"Sigma Evaporation: {sigma_e: .4f}")
    print(f"Sigma Condensation: {sigma_c: .4f}") 
    print(f"Mass flux, J_HKS: {j_hks: .2e} kg/(m^2*s)")
