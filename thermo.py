from thermopack.cpa import cpa
from thermopack.cubic import cubic
from thermopack.saftvrqmie import saftvrqmie
import numpy as np
from collections.abc import Iterable
from thermopack.quantum_cubic import qcubic

R = 8.314

water = cpa('H2O', 'srk')
water.set_tmin(temp=200)
# H2 = saftvrqmie('H2')
H2 = qcubic('H2')
# H2 = cubic("H2", "PR")
H2.set_tmin(temp=2.0)
# print(H2.bubble_temperature(1e5,[1]))
# vg,  = H2.specific_volume(20.483,1e5,[1], H2.VAPPH)
N2 = saftvrqmie('N2')
C8 = saftvrqmie('NC8')

# print(H2.bubble_temperature(1e5,[1]))

M_water = water.compmoleweight(1) * 1e-3  # [kg/mol]
M_H2 = H2.compmoleweight(1) * 1e-3  # [kg/mol]
M_N2 = N2.compmoleweight(1) * 1e-3  # [kg/mol]
M_C8 = C8.compmoleweight(1) * 1e-3  # [kg/mol]

DOF_water = 3
DOF_H2 = 5
DOF_N2 = 5
DOF_C8 = 30 #ish

Tc_water = 647
Tc_N2 = 126.2
Tc_H2 = 33.18



def calc_cp(T, p, phase_flag, eos):
    M = eos.compmoleweight(1) * 1e-3  # [kg/mol]
    if isinstance(T, Iterable):
        cp = np.array([calc_cp(T_val, p, phase_flag, eos)
                       for T_val in T])
        return cp
    else:
        _, Cp_liq = eos.enthalpy(T, p, [1], phase_flag, dhdt=True)
        return Cp_liq/M


def calc_rho(T, p, phase_flag, eos):
    M = eos.compmoleweight(1) * 1e-3  # [kg/mol]
    if isinstance(T, Iterable):
        rho = np.array([calc_rho(T_val, p, phase_flag, eos)
                        for T_val in T])
        return rho
    else:
        specific_volume, = eos.specific_volume(
                T, p, [1], phase_flag)  # [m^3/mol]
        return M/specific_volume

def calc_kappa(T, p, phase_flag, x):
    kappa_vap = 0.017
    kappa_liq = 0.57
    if isinstance(T, Iterable):
        kappa = np.array([calc_kappa(T_val, p, phase_flag, x_val) for T_val, x_val in zip(T, x)])
        return kappa
    
    kappa = kappa_vap if phase_flag == water.VAPPH else kappa_liq
    # kappa = kappa_vap*(1 + 400*x) if phase_flag == water.VAPPH else kappa_liq
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
    # T, _ = eos.get_envelope_twophase(p, [1])
    T_sat, _ = eos.bubble_temperature(p, [1])
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

    x = np.linspace(0, 3e-3, 50)
    T = np.linspace(273, 272, 50)
    kappa = calc_kappa(T, 1e5, water.VAPPH, x)
    plt.plot(kappa, x)
    plt.show()

    print(calc_p_sat(298, water))
    print(calc_T_sat(101325, water))
