from thermopack.cpa import cpa
import numpy as np
from collections.abc import Iterable

water = cpa('H2O', 'SRK')
M = water.compmoleweight(1) * 1e-3  # [kg/mol]


def calc_cp(T, p, phase_flag):
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        cp = np.array([calc_cp(T_val, p_val, phase_flag)
                      for T_val, p_val in zip(T, p)])
        return cp
    else:
        _, Cp_liq = water.enthalpy(T, p, [1], phase_flag, dhdt=True)
        return Cp_liq/M


def calc_rho(T, p, phase_flag):
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        rho = np.array([calc_rho(T_val, p_val, phase_flag)
                       for T_val, p_val in zip(T, p)])
        return rho
    else:
        specific_volume, = water.specific_volume(
            T, p, [1], phase_flag)  # [m^3/mol]
        return M/specific_volume


def calc_kappa(T, p, phase_flag):
    kappa_vap = 0.017
    kappa_liq = 0.57
    if isinstance(T, Iterable) and isinstance(p, Iterable):
        kappa = np.array([calc_kappa(T_val, p_val, phase_flag)
                         for T_val, p_val in zip(T, p)])
        return kappa
    return kappa_vap if (phase_flag == water.VAPPH) else kappa_liq
