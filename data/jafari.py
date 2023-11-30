import numpy as np

p = 882
mdot = 0.39e-4 # average: 2.45e-4
label = "Jafari et al."

q_gas = -10.36
q_liq = 86

T_liq = [5.196304849884527, 5.311778290993072, 5.427251732101617, 5.519630484988453, 5.612009237875289, 5.727482678983834, 5.842956120092379, 5.935334872979215, 6.05080831408776]
x_liq = [-0.018248175182482562, -0.6569343065693438, -1.3381995133819955, -2.0194647201946476, -2.615571776155718, -3.3819951338199514, -4.148418491484185, -4.872262773722627, -5.468369829683699]

T_gas = [5.612009237875289, 6.05080831408776, 6.535796766743649, 6.97459584295612, 7.528868360277136, 8.083140877598153, 8.498845265588916, 9.006928406466512, 9.53810623556582, 9.97690531177829, 10.531177829099308, 11.200923787528868, 13.27944572748268]
x_gas = [0.06690997566909918, 0.6630170316301696, 1.3868613138686126, 2.1532846715328464, 3.090024330900242, 3.856447688564476, 4.708029197080291, 5.559610705596107, 6.624087591240874, 7.47566909975669, 8.753041362530412, 10.072992700729925, 14.075425790754256]

# p = 541
# mdot = 1.87e-4 # average: 2.87e-4
# label = "Jafari et al."

# q_gas = -13.34
# q_liq = 456


# T_liq = [-1.5935334872979214, -1.2702078521939955, -0.9468822170900693, -0.6466512702078522, -0.3233256351039261, 0.023094688221709007, 0.39260969976905313, 0.7621247113163973, 1.0392609699769053, 1.3163972286374135, 1.6628175519630486]
# x_liq = [-0.018248175182482562, -0.35888077858880774, -0.8272506082725064, -1.2530413625304138, -1.6788321167883216, -2.1897810218978107, -2.74330900243309, -3.3394160583941606, -3.8503649635036497, -4.489051094890511, -5.170316301703163]

# T_gas = [-1.1085450346420322, -0.7852193995381063, -0.3695150115473441, 0.06928406466512702, 0.4618937644341802, 0.9468822170900693, 1.4780600461893765, 2.0554272517321017, 2.655889145496536]
# x_gas = [0.1520681265206809, 0.4501216545012161, 0.833333333333333, 1.3017031630170308, 1.7274939172749386, 2.2384428223844273, 2.9622871046228703, 3.728710462287104, 4.537712895377128]