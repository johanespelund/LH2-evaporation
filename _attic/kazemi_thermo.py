import numpy as np

# Constants
R = 8.314  # J/(mol K)

# Liquid phase properties
def mu_liquid(T):
    """Dynamic viscosity of liquid (Pa.s) as a function of temperature (K)"""
    return 2.71 * np.exp(-0.0352 * T)

def rho_liquid(T):
    """Density of liquid (kg/m^3) as a function of temperature (K)"""
    return (999.84 + 16.945 * T - 7.99e-3 * T**2 - 46.17e-6 * T**3 + 105.56e-9 * T**4 - 280.54e-12 * T**5)/(1 + 16.88e-3 * T)

def sigma_liquid(T):
    """Surface tension of liquid (N/m) as a function of temperature (K)"""
    return 1e-3 * (71.18 - 0.1435 * T + 16.88 * T**2)

def kappa_liquid(T):
    """Thermal conductivity of liquid (W/m K) as a function of temperature (K)"""
    return ((T/228) - 1)**0.18

def psat_liquid(T):
    """Saturation pressure (Pa) as a function of temperature (K)"""
    return 611.2 * np.exp(1045.85115 - 21394.66626 / T + 1.09697 * T - 1.300374 * 10**(-3) * T**2 +
                          7.747299 * 10**(-7) * T**3 - 2.1649 * 10**(-12) * T**4 - 211.3896 * np.log(T))

def cp_liquid(T):
    """Specific heat capacity of liquid (J/kg K) as a function of temperature (K)"""
    T = T - 273.15 # Polynomial expects temperature in deg. C
    return 4213.1 - 2.92247 * T + 0.14372 * T**2 - 5e-4 * T**3 - 1e-4 * T**4 + 2e-5 * T**5

# Vapor phase properties
def mu_vapor(T):
    """Dynamic viscosity of vapor (Pa.s) as a function of temperature (K)"""
    return (1e-4 * np.sqrt(T) * (0.647 * (1 + 0.282 * T / (132 + T))))

def cp_vapor(T):
    """Specific heat capacity of vapor (J/kg K) as a function of temperature (K)"""
    return 1875.711 - 3.465e-1 * T - 5.199e-4 * T**2 + 7.240e-6 * T**3

def kappa_vapor(T):
    """Thermal conductivity of vapor (W/m K) as a function of temperature (K)"""
    return 0.0088 - 1e-5 * T + 1.4e-7 * T**2

M_water_vapor = 0.018

def rho_vapor(T, p):
    """Density of vapor (kg/m^3) as a function of temperature (K)"""
    return p * M_water_vapor / (R * T)

# Solid phase properties (constants)
solid_properties = {
    "copper": {"k": 386, "rho": 8930, "cp": 385},
    "aluminum": {"k": 230, "rho": 2700, "cp": 900},
    "borosilicate_glass": {"k": 1.14, "rho": 2210, "cp": 730}
}

def k_solid(material):
    """Thermal conductivity of solid (W/m K)"""
    return solid_properties[material]["k"]

def rho_solid(material):
    """Density of solid (kg/m^3)"""
    return solid_properties[material]["rho"]

def cp_solid(material):
    """Specific heat capacity of solid (J/kg K)"""
    return solid_properties[material]["cp"]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    T = np.linspace(260, 300)
    plt.plot(T, kappa_vapor(T))
    plt.show()