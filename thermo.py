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
        return M_waterjk/specific_volume

def smooth_transition(x, x_start, x_end, transition_thickness, scale):
    # If x is before the transition starts, return 1
    if x <= x_start:
        return 1
    # If x is after the transition ends, return scale
    elif x >= x_start + transition_thickness:
        return scale
    # If x is within the transition region, apply a smooth change
    else:
        # Calculate the transition using a sigmoid function
        # The midpoint of the transition will be at x_start + transition_thickness / 2
        midpoint = x_start + transition_thickness / 2
        # The steepness of the sigmoid is inversely proportional to the thickness
        steepness = 10 / transition_thickness
        # Calculate the sigmoid value
        sigmoid_value = 1 / (1 + np.exp(-steepness * (x - midpoint)))
        # Scale the transition between 1 and the scale value
        return (1 - sigmoid_value) + sigmoid_value * scale

def calc_kappa(T, p, phase_flag, x):
    kappa_vap = 0.017
    kappa_liq = 0.57
    if isinstance(T, Iterable):
        kappa = np.array([calc_kappa(T_val, p, phase_flag, x_val) for T_val, x_val in zip(T, x)])
        return kappa

    kappa = kappa_vap if phase_flag == water.VAPPH else kappa_liq

    # if phase_flag == water.LIQPH:
    #     return kappa * (1 - 0.01*x*5e3)

    if x <= -2.5e-3 and x >= -5e-3 and phase_flag == water.LIQPH:
        # Calculate a smooth transition factor for kappa in the range of x.
        transition_factor = smooth_transition(x, -2.5e-3, -5e-3, 1e-3, 2)
        kappa *= (1 + (-x - 2.5e-3)**2*800000) #transition_factor
        return kappa

    return kappa




# def calc_kappa(T, p, phase_flag, x):
#     kappa_vap = 0.017
#     kappa_liq = 0.57
#     if isinstance(T, Iterable):
#         kappa = np.array([calc_kappa(T_val, p, phase_flag, x_val)
#                           for T_val, x_val in zip(T, x)])
#         return kappa
#     kappa = kappa_vap if (phase_flag == water.VAPPH) else kappa_liq
#     if x < 0 and x > -2.5e-3:
#         kappa *= 1e-2  # (x + 2.5e-3)*2*2.5e3
#         # print(x, kappa)
#     return kappa


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
