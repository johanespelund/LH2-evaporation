from thermopack.cpa import cpa
from thermopack.saftvrqmie import saftvrqmie
import numpy as np
from collections.abc import Iterable

R = 8.314

water = cpa('H2O')
H2 = saftvrqmie('H2')
M_water = water.compmoleweight(1) * 1e-3  # [kg/mol]
M_H2 = H2.compmoleweight(1) * 1e-3  # [kg/mol]


def calc_cp(T, p, phase_flag, eos):
    if isinstance(T, Iterable):
        cp = np.array([calc_cp(T_val, p, phase_flag, eos)
                       for T_val in T])
        return cp
    else:
        _, Cp_liq = eos.enthalpy(T, p, [1], phase_flag, dhdt=True)
        return Cp_liq/M_water


def calc_rho(T, p, phase_flag, eos):
    if isinstance(T, Iterable):
        rho = np.array([calc_rho(T_val, p, phase_flag, eos)
                        for T_val in T])
        return rho
    else:
        specific_volume, = eos.specific_volume(
                T, p, [1], phase_flag)  # [m^3/mol]
        return M_water/specific_volume

def calc_kappa(T, p, phase_flag, x):
    kappa_vap = 0.017
    kappa_liq = 0.57
    if isinstance(T, Iterable):
        kappa = np.array([calc_kappa(T_val, p, phase_flag, x_val) for T_val, x_val in zip(T, x)])
        return kappa

    kappa = kappa_vap*(1 + 400*x) if phase_flag == water.VAPPH else kappa_liq
    # if x <= -2.5e-3 and x >= -5e-3 and phase_flag == water.LIQPH:
    #     kappa *= (1 + (-x - 2.5e-3)**2*800000) #transition_factor
    #     return kappa

    return kappa



def calc_dHvap(T_l, T_g, p, eos):
    M = eos.compmoleweight(1) * 1e-3  # [kg/mol]
    hg, = eos.enthalpy(T_g, p, [1], eos.VAPPH)
    hl, = eos.enthalpy(T_l, p, [1], eos.LIQPH)
    return (hg - hl)/M


def calc_p_sat(T, eos):
    if isinstance(T, Iterable):
        p_sat = np.array([calc_p_sat(T_val, eos)
                          for T_val in T])
        return p_sat
    p_sat, _ = eos.bubble_pressure(T, [1])
    return p_sat


def calc_T_sat(p, eos):
    T_sat, _ = water.bubble_temperature(p, [1])
    return T_sat


def calc_Ceq(T, eos):
    if isinstance(T, Iterable):
        C_eq = np.array([calc_Ceq(T_val, eos)
                         for T_val in T])
        return C_eq
    psat = calc_p_sat(T, eos)
    vg, = eos.specific_volume(
            T, psat, [1], eos.VAPPH)
    C_eq = 1/vg  # mol/m^3
    return C_eq


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from thermo import water

    x = np.linspace(-5e-3, 0, 50)
    T = np.linspace(273, 272, 50)
    kappa = calc_kappa(T, 1e5, water.LIQPH, x)
    plt.plot(kappa, x)
    plt.show()

    print(calc_p_sat(298, water))
    print(calc_T_sat(101325, water))

# import CoolProp.CoolProp as CP
# import numpy as np
# from collections.abc import Iterable

# # Constants
# R = 8.314

# # Function to calculate specific heat capacity
# def calc_cp(T, p, fluid):
#     if isinstance(T, Iterable):
#         cp = np.array([calc_cp(T_val, p, fluid) for T_val in T])
#         return cp
#     else:
#         return CP.PropsSI('Cpmass', 'T', T, 'P', p, fluid)


# # Function to calculate density
# def calc_rho(T, p, fluid):
#     if isinstance(T, Iterable):
#         rho = np.array([calc_rho(T_val, p, fluid) for T_val in T])
#         return rho
#     else:
#         return CP.PropsSI('Dmass', 'T', T, 'P', p, fluid)


# # Function to calculate thermal conductivity
# def calc_kappa(T, p, fluid):
#     if isinstance(T, Iterable):
#         kappa = np.array([calc_kappa(T_val, p, fluid) for T_val in T])
#         return kappa
#     else:
#         return CP.PropsSI('conductivity', 'T', T, 'P', p, fluid)


# # Function to calculate enthalpy of vaporization
# def calc_dHvap(T, p, fluid):
#     h_g = CP.PropsSI('Hmass', 'T', T, 'P', p, 'Q', 1, fluid)  # Vapor
#     h_l = CP.PropsSI('Hmass', 'T', T, 'P', p, 'Q', 0, fluid)  # Liquid
#     return h_g - h_l


# # Function to calculate saturation pressure
# def calc_p_sat(T, fluid):
#     if isinstance(T, Iterable):
#         p_sat = np.array([calc_p_sat(T_val, fluid) for T_val in T])
#         return p_sat
#     else:
#         return CP.PropsSI('P', 'T', T, 'Q', 0, fluid)


# # Function to calculate saturation temperature
# def calc_T_sat(p, fluid):
#     return CP.PropsSI('T', 'P', p, 'Q', 0, fluid)


# # Function to calculate equilibrium concentration
# def calc_Ceq(T, fluid):
#     if isinstance(T, Iterable):
#         C_eq = np.array([calc_Ceq(T_val, fluid) for T_val in T])
#         return C_eq
#     else:
#         psat = calc_p_sat(T, fluid)
#         v_g = CP.PropsSI('V', 'T', T, 'Q', 1, fluid)  # Q = 1 for vapor phase
#         return 1 / v_g



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np

#     x = np.linspace(-5e-3, 0, 50)
#     T = np.linspace(273.25, 450, 50)
#     kappa = calc_rho(T, 1e5, "Water")
#     plt.plot(T, kappa)
#     plt.show()

#     print(calc_p_sat(280, "Water"))
#     print(calc_T_sat(1e5, "Water"))
