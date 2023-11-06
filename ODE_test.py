from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from thermopack.cpa import cpa


water = cpa('H2O', 'SRK')
M = water.compmoleweight(1) * 1e-3  # [kg/mol]

p0 = 5000
T0 = 273.15 - 7
L = -10e-3

# Define the ODE system
def odes(x, y, mdot, kappa, phase_flag=water.LIQPH):
    h, q = y
    p = p0 + 9.81*x*1000
    cp = cp_of_T(h_to_T(h, p), p, phase_flag=phase_flag)  # You need to define cp_of_T and h_to_T
    dhdx = -cp * q / kappa
    dqdx = -mdot * dhdx
    return [dhdx, dqdx]


# Constants
mdot = 3.17e-4  # Mass flow rate
kappa = 0.5 # Thermal conductivity

# Define the function for cp(T)
def cp_of_T(T, p, phase_flag):
    _, Cp_liq = water.enthalpy(T, p, [1], phase_flag, dhdt=True)
    return Cp_liq/M

# Define the function to convert h to T
def h_to_T(h, p):
    # You need to provide this function
    flsh = water.two_phase_phflash(p, [1], h*M)
    return flsh.T


# Boundary conditions at x = L
# flsh = water.two_phase_tpflash(T0+5, p0 + 9.81*0.1e-3*1000, [1])
# print(flsh)
# exit()
h_L = water.enthalpy(T0, p0, [1], water.LIQPH)[0]/M
q_L = 600  # Specify your q_L

# Initial guesses for h and q at x = L
initial_guess = [h_L, q_L]

# Integrate from L to 0
solution = solve_ivp(lambda x, y: odes(x, y, mdot, kappa),
                     [0, L], initial_guess, t_eval=np.linspace(0, L, 200))#, method='Radau')  # Radau is good for stiff ODEs

# Extract the solution
x_values = solution.t
h_values = solution.y[0]
q_values = solution.y[1]

# Check if the solver reached a solution

T = np.zeros(solution.t.size)

# energy_flux = mdot*h_values + q_values

# energy_flux_grad = np.gradient(energy_flux, x_values) - 1000000
# print(energy_flux_grad/energy_flux)

if solution.success:
    for i in range(T.size):
        T[i] = h_to_T(solution.y[0][i], p0)
    plt.plot(T - 273.15, solution.t, label='Temperature')
    plt.ylabel('Position (m)')
    plt.xlabel('Temperature (K)')
    plt.legend()
    plt.title('Temperature Distribution')
else:
    print("The solver did not converge to a solution")

p0 = 300
T0 = 273.15 - 6.3
L = 30e-3

kappa = 0.017 # Thermal conductivity

# Define the ODE system
def odes(x, y, mdot, kappa, phase_flag=water.LIQPH):
    h, q = y
    p = p0
    cp = cp_of_T(h_to_T(h, p), p, phase_flag=phase_flag)  # You need to define cp_of_T and h_to_T
    dhdx = -cp * q / kappa
    dqdx = mdot * dhdx
    return [dhdx, dqdx]

# Boundary conditions at x = L
h_L = water.enthalpy(T0, p0, [1], water.VAPPH)[0]/M
q_L = -15.84  # Specify your q_L

# Initial guesses for h and q at x = L
initial_guess = [h_L, q_L]

# Integrate from L to 0
solution = solve_ivp(lambda x, y: odes(x, y, mdot, kappa, phase_flag=water.VAPPH),
                     [0, L], initial_guess, t_eval=np.linspace(0, L, 200))#, method='Radau')  # Radau is good for stiff ODEs

# Extract the solution
x_values = solution.t
h_values = solution.y[0]
q_values = solution.y[1]

# Check if the solver reached a solution

T = np.zeros(solution.t.size)

energy_flux = mdot*h_values + q_values

energy_flux_grad = np.gradient(energy_flux, x_values)
print(energy_flux_grad/energy_flux)

if solution.success:
    for i in range(T.size):
        T[i] = h_to_T(solution.y[0][i], p0)
    plt.plot(T - 273.15, solution.t, label='Temperature')

plt.xlim((-10, 22))
plt.ylim((-10e-3, 35e-3))
plt.show()
