# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# L = 0.1         # [m]
# A_c = 1         # Cross-sectional area [m^2]
# nx = 100        # [-]


# U_in = 0.01     # [m/s]
# p_atm = 1.035e5 # [Pa]
# T_in = 300      # [K]

# c_p = 700               # [J/(kg*K)]
# kappa = 0.003           # [W/(m*K)]
# R = 287.05              # Individual gas constant for air [J/(kg*K)]
# q_dot = 700             # Heating source [W/m^3]  
# rho_in = p_atm/(R*T_in)  # [kg/m^3] 

# x = np.linspace(0, L, nx)
# rho = np.ones(x.size)*rho_in
# u = np.ones(x.size)*U_in
# h = np.ones(x.size)*T_in*c_p
# p = np.ones(x.size)*p_atm
# T = np.ones(x.size)*T_in 

# print(rho_in)

# for iter in range(150):
#     for i in range(nx):
#         dTdx = np.gradient(T, x)
#         d2Tdx2 = np.gradient(T, x)
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt



# from thermopack.cubic import cubic

# eos = cubic('H2', 'PR')
# eos.set_tmin(20)

# x = [1]
# p0 = 2e5
# T0 = 25

# print(
#     eos.two_phase_tpflash(T0, p0, x),
#     eos.enthalpy(T0, p0, x, eos.LIQPH)[0] -
#     eos.enthalpy(T0, p0, x, eos.VAPPH)[0]
    
#     # eos.specific_volume()
# )

# T, p = eos.get_envelope_twophase(0.95e5, x) # arrays of temperature and pressure for phase envelope, starting at 1 bar.
# plt.fill_between(T, p*1e-5)
# plt.plot(T, p*1e-5) # Tp-projection of phase envelope
# plt.plot(T0, p0*1e-5, 'kx')
# plt.grid()
# plt.show()

# exit()

# Constants
R = 287.05  # J/(kg*K), specific gas constant for air
c_p = 1005  # J/(kg*K), specific heat at constant pressure for air
q_0 = 1000  # W/m, constant heat source term
kappa = 0.003           # [W/(m*K)]

# Boundary conditions
T_inlet = 300  # K, inlet temperature
u_inlet = 0.1   # m/s, inlet velocity
p_outlet = 101325  # Pa, outlet pressure

# Inlet density from ideal gas law
rho_inlet = p_outlet / (R * T_inlet)

# Inlet mass flow rate
m_dot_inlet = rho_inlet * u_inlet  # kg/s

def density(x, p, T, phase):
    if phase == "liq":
        return np.ones(p.size)*10 #0*np.ones(p.size)*10*np.abs(x) + p / (R * T)
    else:
        return p / (R * T)

# Define the ODEs to be solved
def odes(x, vars):
    T, p, u, dudx, d2Tdx2 = vars  # Temperature and pressure
    rho = density(x, p, T)
    print(rho)

    # ODEs
    # q_0 = 1000*(T - 298)
    dTdx = (-kappa*d2Tdx2 + q_0)/(rho*u*c_p) #q_0 / (rho * u * c_p)
    dpdx = -rho*u*dudx
    return dTdx, dpdx

def transform(x, width):
    return np.tanh(x / width)

# Spatial discretization
L = 0.1  # Length of the pipe in meters
n_points = 25 # Number of discretization points
# Create a uniform distribution of points in the transformed space
width = L / np.arctanh(0.99)  # width is chosen to avoid the tanh function reaching the limits -1 and 1

non_linear_points = np.linspace(-np.sqrt(L/2), 0, n_points)**2
# non_linear_points = np.linspace(-L/2, 0, n_points) # np.linspace(-np.sqrt(L/2), 0, n_points)**2
print(non_linear_points)

dense_points_right = np.flip(non_linear_points)
dense_points_left = -non_linear_points

dense_points = np.concatenate((dense_points_left, dense_points_right))
x_values = dense_points #np.linspace(-L/2, L/2, n_points) # 
# print(x_values)
# exit()
# Initial conditions for T and p
initial_conditions = [T_inlet, p_outlet]

# We'll store the results here
Jq_values = np.zeros(n_points*2)
T_values = np.ones(n_points*2)*T_inlet
h_values = np.ones(n_points*2)*T_inlet*c_p
p_values = np.ones(n_points*2)*p_outlet
rho_values = np.ones(n_points*2)*rho_inlet
u_values = np.ones(n_points*2)*m_dot_inlet / rho_inlet

# Initial values
T_values[0] = T_inlet
u_values[0] = m_dot_inlet / rho_inlet  # Velocity from the mass conservation
p_values[-1] = p_outlet  # We're integrating backwards for pressure

p_old = np.empty(n_points*2)
u_old = np.empty(n_points*2)
T_old = np.empty(n_points*2)
rho_old = np.empty(n_points*2)
h_old = np.empty(n_points*2)


# Initialize dictionaries
u = {}
h = {}
p = {}
rho = {}
T = {}
x = {}

# Split the arrays at n_points and assign to the dictionaries
x['liq'] = x_values[:n_points]
x['gas'] = x_values[n_points:]

u['liq'] = u_values[:n_points]
u['gas'] = u_values[n_points:]

h['liq'] = h_values[:n_points]
h['gas'] = h_values[n_points:]

p['liq'] = p_values[:n_points]
p['gas'] = p_values[n_points:]

rho['liq'] = rho_values[:n_points]
rho['gas'] = rho_values[n_points:]

T['liq'] = T_values[:n_points]
T['gas'] = T_values[n_points:]

for _ in range(10000):
    p_old[:] = p_values[:]
    u_old[:] = u_values[:]
    T_old[:] = T_values[:]
    rho_old[:] = rho_values[:]
    for phase in ['liq', 'gas']:
        dudx = np.gradient(u[phase], x[phase])
        dhdx = np.gradient(h[phase], x[phase])
        drhodx = np.gradient(rho[phase], x[phase])
        
        
        for i in range(n_points):
            T[phase][i] = h[phase][i]/c_p
        dTdx = np.gradient(T[phase], x[phase])
        Jq = -kappa*dTdx
        rho[phase] = density(x[phase], p[phase], T[phase], phase)
        
        q_0 = 1000*(T[phase] - 298)
        
        h[phase][1:] = h[phase][:-1] + ((-np.gradient(Jq, x[phase]) + q_0)/(rho[phase]*u[phase]))[:-1]*(x[phase][1:] - x[phase][:-1])
        p[phase][:-1] = p[phase][1:] + (rho[phase]*u[phase]*dudx)[1:]*(x[phase][:-1] - x[phase][1:])
        # p[phase][1:] = p[phase][:-1] + (rho[phase]*u[phase]*dudx)[:-1]*(x[phase][1:] - x[phase][:-1])
        u[phase] = m_dot_inlet/rho[phase]
        
        # Assuming u['liq'] and u['gas'] have been modified and you want to reflect these changes in u_values
        u_values = np.concatenate((u['liq'], u['gas']), axis=0)

        # Similarly, for other properties
        h_values = np.concatenate((h['liq'], h['gas']), axis=0)
        p_values = np.concatenate((p['liq'], p['gas']), axis=0)
        rho_values = np.concatenate((rho['liq'], rho['gas']), axis=0)
        T_values = np.concatenate((T['liq'], T['gas']), axis=0)


        print(np.mean(np.abs(p_old - p_values)))
        print(np.mean(np.abs(u_old - u_values)))
        print(np.mean(np.abs(T_old - T_values)))
        print(np.mean(np.abs(rho_old - rho_values)))
        print("")
        u['gas'][0] = u['liq'][-1]
        p['liq'][-1] = p['gas'][0]
        # p['gas'][0] = p['liq'][-1]
        h['gas'][0] = h['liq'][-1]
        T['gas'][0] = T['liq'][-1]
    
    # Assuming u['liq'] and u['gas'] have been modified and you want to reflect these changes in u_values
    u_values = np.concatenate((u['liq'], u['gas']), axis=0)

    # Similarly, for other properties
    h_values = np.concatenate((h['liq'], h['gas']), axis=0)
    p_values = np.concatenate((p['liq'], p['gas']), axis=0)
    rho_values = np.concatenate((rho['liq'], rho['gas']), axis=0)
    T_values = np.concatenate((T['liq'], T['gas']), axis=0)


# for iter in range(10):
#     # Integrate the ODEs using a simple Euler method
#     p_old[:] = p_values[:]
#     u_old[:] = u_values[:]
#     T_old[:] = T_values[:]
#     rho_old[:] = rho_values[:]
#     for i in range(1, n_points):
#         dx = x_values[i] - x_values[i - 1]
#         dudx = np.gradient(u_values, x_values)
#         d2Tdx2 = np.gradient(np.gradient(T_values, x_values))
#         dTdx, dpdx = odes(0.5*(x_values[i-1] + x_values[i]), [T_values[i - 1], p_values[n_points - i], u_values[n_points - i], dudx[n_points - i], d2Tdx2[i-1]])
#         T_values[i] = T_values[i - 1] + dTdx * dx
#         p_values[n_points - i - 1] = p_values[n_points - i] - dpdx * dx
#     # if iter == 9:
#     #     exit()
#     # Calculate velocity and density from T and p
#     for i in range(n_points):
#         rho_values[i] = density(x_values[i], p_values[i], T_values[i])
#         u_values[i] = m_dot_inlet / rho_values[i]

#     print(np.mean(np.abs(p_old - p_values)))
#     print(np.mean(np.abs(u_old - u_values)))
#     print(np.mean(np.abs(T_old - T_values)))
#     print(np.mean(np.abs(rho_old - rho_values)))
#     print("")

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x_values, T_values, 'r')
plt.title('Temperature along the pipe')
plt.xlabel('x (m)')
plt.ylabel('Temperature (K)')

plt.subplot(2, 2, 2)
plt.plot(x_values, p_values-p_outlet, 'b-o')
plt.title('Pressure along the pipe')
plt.xlabel('x (m)')
plt.ylabel('Pressure (bar)')

plt.subplot(2, 2, 3)
plt.plot(x_values, rho_values, 'g')
plt.title('Density along the pipe')
plt.xlabel('x (m)')
plt.ylabel('Density (kg/m^3)')

plt.subplot(2, 2, 4)
plt.plot(x_values, u_values, 'k')
plt.title('Velocity along the pipe')
plt.xlabel('x (m)')
plt.ylabel('Velocity (m/s)')

# plt.subplot(2, 2, 2)
# plt.plot(x_values, p_values / 1e5, 'b')
# plt.title('Pressure along the pipe')
# plt.xlabel('x (m)')
# plt.ylabel('Pressure (bar)')

# plt.subplot(2, 2, 3)
# plt.plot(x_values, rho_values*u_values, 'g')
# plt.title('Mass flux')
# plt.xlabel('x (m)')
# plt.ylabel('Density (kg/m^3)')

# plt.subplot(2, 2, 4)
# plt.plot(x['liq'], rho['liq']*u['liq']*np.gradient(u['liq'], x['liq']) + np.gradient(p['liq'], x['liq']), 'ko')
# plt.plot(x['gas'], rho['gas']*u['gas']*np.gradient(u['gas'], x['gas']) + np.gradient(p['gas'], x['gas']), 'r.')
# plt.title('Momentum')
# plt.xlabel('x (m)')
# plt.ylabel('Velocity (m/s)')

plt.tight_layout()
plt.show()
