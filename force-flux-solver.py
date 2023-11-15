import numpy as np
import NET
import thermo
from scipy.optimize import fsolve

eos = thermo.water
M = thermo.M_water
R = 8.314


Tvap = 273 + 2
qvap = -50.36
p = thermo.calc_p_sat(Tvap, eos) - 5


J0 = 0
Tl0 = Tvap


def deltaT_inv(q_g, J, rqq_sg, rqmu_sg):
    return rqq_sg*q_g + rqmu_sg*J


def p_by_psat(q_g, J, rmumu_sg, rqmu_sg):
    return rqmu_sg*q_g + rmumu_sg*J


def func(x):
    _Tl, J = x
    p_sat = thermo.calc_p_sat(_Tl, eos)
    C_eq = thermo.calc_Ceq(_Tl, eos)

    r_qq = NET.R_qq(C_eq, _Tl, R, M)
    r_qmu = NET.R_qmu(C_eq, _Tl, R, M)
    r_mumu = NET.R_mumu(C_eq, _Tl, R, M)

    r1 = (1/Tvap) - (1/_Tl) - deltaT_inv(qvap, J, r_qq, r_qmu)
    r2 = -R*np.log(p/p_sat) - \
        p_by_psat(qvap, J, r_mumu, r_qmu)
    return np.array([r1, r2])


res = fsolve(func, np.array([Tl0, J0]))
Tliq, J = res
mdot = J/M
print(Tliq, Tvap)
print(f"{mdot=}")
print(func([Tliq, J]))
