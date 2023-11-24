import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from thermopack.cubic import cubic
from thermopack.cpa import cpa
from thermopack import thermo

from _attic.density_func import density
from dataclasses import dataclass

water = cpa('H2O', 'SRK')

N = 20
L = 1e-3

x = np.linspace(0, L, N)

non_linear_points = np.linspace(-np.sqrt(L/2), 0, N)**2
dense_points_right = np.flip(non_linear_points)
dense_points_left = -non_linear_points

dense_points = np.concatenate((dense_points_left[:-1], dense_points_right))
# x = dense_points
N = x.size

x_vap = x
x_liq = -np.flip(x)

# Constants
R = 287.05  # J/(kg*K), specific gas constant for air
c_p = 1805  # J/(kg*K), specific heat at constant pressure for air
kappa = 0.62           # [W/(m*K)]
liquid_density = 1000

htc = 0.01
T_ref = 300

# Boundary conditions
m_dot = 4.70e-4            # kg/(m^2*s), mass flux
p_inlet = 561     # Pa, inlet pressure
T_sat, p_sat = water.get_envelope_twophase(p_inlet, [1])
T_inlet = 273.15 #T_sat[0]           # K, inlet temperature
q_l = 546.26            # [W/m^2]
print(T_inlet)

# Inlet density from ideal gas law
rho_inlet = liquid_density #p_inlet / (R * T_inlet)
u_inlet = m_dot/rho_inlet  # m/s

h_inlet =  water.enthalpy(T_sat[0], p_sat[0], [1], water.LIQPH)[0]


# print(h_inlet)
# flsh = water.two_phase_phflash(p_sat[0], [1], h_inlet + 0*water.compmoleweight(1)*1e-3)
# print(h_inlet, flsh)
# print(p_sat[0], T_sat[0]-273.15)


class Phase:
    def __init__(self, x, N, eos, phase_flag):
        self.x = x
        self.N = N
        self.eos = eos
        
        self.dx = np.gradient(self.x)
        self.M = self.eos.compmoleweight(1) * 1e-3  # [kg/mol]
        
        self.phase_flag = phase_flag
        self.u = np.ones(N) * u_inlet
        self.rho = np.ones(N) * rho_inlet
        self.h = np.ones(N) * h_inlet/self.M
        self.T = np.ones(N) * T_inlet
        self.p = np.ones(N) * p_inlet
        self.q = np.zeros(N)
        self.fields = [self.u, self.rho, self.h, self.T, self.p, self.q]
        
        # Initialize gradients
        self.dudx = np.zeros(N)
        self.dhdx = np.zeros(N)
        self.dTdx = np.zeros(N)
        self.dpdx = np.zeros(N)
    
    # Calculation of gradients
    def calc_dhdx(self):
        self.dhdx = (-np.gradient(self.q, self.x, edge_order=2) + self.Q_dot(htc, T_ref))/m_dot 
        return self.dhdx
    
    def calc_dpdx(self):
        self.dpdx = -m_dot*self.dudx
        return -m_dot*self.dudx
    
    def update_fields(self, backwards=True):
        self.calc_dhdx()
        self.calc_dpdx()
        
        if backwards:
            self.h[:-1] = self.h[1:] - self.dhdx[1:]*self.dx[1:]
            self.p[:-1] = self.p[1:] - self.dpdx[1:]*self.dx[1:]
        else:
            self.h[1:] = self.h[:-1] + self.dhdx[:-1]*self.dx[:-1]
            self.p[1:] = self.p[:-1] + self.dpdx[:-1]*self.dx[:-1]


        # self.T[:-1] = self.T[1:] - self.dx[1:]*    
            
        # Update temperature, density and velocity
        # self.T = self.h/c_p
        self.q[1:] = (-self.kappa()*np.gradient(self.T, self.x, edge_order=2))[1:]
        self.calc_temp()
        self.calc_density()
        self.u = m_dot/self.rho
        self.dudx = np.gradient(self.u, self.x, edge_order=2)
        
    def Q_dot(self, htc, T_ref):
        return 0#htc*(T_ref - self.T)
    
    def calc_density(self):
        for i in range(self.N):
            specific_volume, = self.eos.specific_volume(self.T[i], self.p[i], [1], self.phase_flag) # [m^3/mol]
            self.rho[i] = self.M/specific_volume
    
    def kappa(self):
        if self.phase_flag == water.VAPPH:
            return 0.017
        else:
            return 0.55
    
    def calc_temp(self):
        print(self.h)
        for i in range(self.N):
            flsh = self.eos.two_phase_phflash(self.p[i], [1], self.h[i]*self.M)
            self.T[i] = flsh.T

liq = Phase(x=x_liq, N=x_liq.size, eos=water, phase_flag=water.LIQPH)
vap = Phase(x=x_vap, N=x_vap.size, eos=water, phase_flag=water.VAPPH)

N_iter = 50

for _ in range(N_iter):
    for _ in range(1):
        liq.update_fields()
    # for i in range(len(liq.fields)):
    #     # Make everything except enthalpy continuous
    #     if i != 2:
    #         vap.fields[i][0] = liq.fields[i][-1]
          
    # dH_vap = (water.enthalpy(vap.T[0], vap.p[0], [1], water.VAPPH)[0] - 
    #           water.enthalpy(vap.T[0], vap.p[0], [1], water.LIQPH)[0])/vap.M
    # vap.h[0] = liq.h[-1] + dH_vap
    # liq.q[-1] =   m_dot*dH_vap + vap.q[0]
    # vap.calc_temp()
    # vap.T[1] = vap.T[0] + vap.dx[0]*vap.q[0]/vap.kappa()
    
    # for _ in range(20):
    #     vap.update_fields(backwards=False)
    
        
fig, axes = plt.subplots(2, 2)
a = axes.flatten()

for phase in [liq]:
    a[0].plot(phase.T - 273.15, phase.x)
    a[1].plot(phase.p, phase.x) 
    a[2].plot(phase.u, phase.x)
    a[3].plot(phase.rho, phase.x)
    
a[0].set_xlabel("Temperature [K]")
a[1].set_xlabel("Pressure [bar]")
a[2].set_xlabel("Velocity [m/s]")
a[3].set_xlabel("Density [kg/m^3]")

for ax in a:
    ax.grid()

fig.tight_layout()
plt.show()
