from thermopack.cpa import cpa
from thermopack.saftvrqmie import saftvrqmie
import numpy as np
from collections.abc import Iterable

water = cpa('H2O')
H2 = saftvrqmie('H2')
M_water = water.compmoleweight(1) * 1e-3  # [kg/mol]
M_H2 = H2.compmoleweight(1) * 1e-3  # [kg/mol]


def calc_cp(T, p, phase_flag, eos):
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        cp = np.array([calc_cp(T_val, p_val, phase_flag, eos)
                      for T_val, p_val in zip(T, p)])
        return cp
    else:
        _, Cp_liq = eos.enthalpy(T, p, [1], phase_flag, dhdt=True)
        return Cp_liq/M_waterjk


def calc_rho(T, p, phase_flag, eos):
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        rho = np.array([calc_rho(T_val, p_val, phase_flag, eos)
                       for T_val, p_val in zip(T, p)])
        return rho
    else:
        specific_volume, = eos.specific_volume(
            T, p, [1], phase_flag)  # [m^3/mol]
        return M_waterjk/specific_volume


def calc_kappa(T, p, phase_flag):
    kappa_vap = 0.017
    kappa_liq = 0.57
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        kappa = np.array([calc_kappa(T_val, p_val, phase_flag)
                         for T_val, p_val in zip(T, p)])
        return kappa
    return kappa_vap if (phase_flag == water.VAPPH) else kappa_liq


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
    print(calc_p_sat(298, water))
    print(calc_T_sat(101325, water))
