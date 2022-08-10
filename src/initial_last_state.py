import numpy as np
import pandas as pd
from sympy import Symbol, nsolve, sqrt, atan2

# Wrap radians to [−pi pi]
def wrapToPi(d):
    # Wrap to [0..2*pi]
    d = d % (2 * np.pi)
    # Wrap to [-pi..pi]
    if d > np.pi:
        d -= (2 * np.pi)
    return d


radar1 = pd.read_csv('../dataset/radar1.csv', header=None).to_numpy()
radar2 = pd.read_csv('../dataset/radar2.csv', header=None).to_numpy()

for i in range(0, len(radar1)):
    radar1[i][1] = wrapToPi(radar1[i][1])
    radar2[i][1] = wrapToPi(radar2[i][1])



################### Radar 1- Initial Position ##########################
Xt = Symbol('Xt')
Yt = Symbol('Yt')
d1 = radar1[0, 0]
phi1 = radar1[0, 1]
eq1 = sqrt(Xt ** 2 + Yt ** 2) - d1
# θ = 0 for initialization
eq2 = atan2(Yt, Xt) - phi1
robot_state = nsolve((eq1, eq2), (Xt, Yt), (-1, 1))
print(robot_state)
#afairesi y kai X Robot


################### Radar 2 - Initial Position ##########################
X_s = np.array([[10 - 6.3], [3.85], [0.0]])
d2 = radar2[0, 0]
phi2 = radar2[0, 1]
eq1 = sqrt((Xt) ** 2 + (Yt) ** 2) - d2
# θ = 0 for initialization
eq2 = atan2(Yt, Xt) - phi2
robot_state = nsolve((eq1, eq2), (Xt, Yt), (-1, 1))
print(robot_state)
#afairesi y kai X Robot



################### Radar 1- Last Position ##########################
Xt = Symbol('Xt')
Yt = Symbol('Yt')
d1_last = radar1[-1, 0]
phi1_last = radar1[-1, 1]
eq1 = sqrt(Xt ** 2 + Yt ** 2) - d1_last
# θ = 0 for initialization
eq2 = atan2(Yt, Xt) - phi1_last
robot_state = nsolve((eq1, eq2), (Xt, Yt), (-1, 1))
print(robot_state)
#[3.12460298067453], [5.55216149019099]]
#afairesi y kai X Robot

################### Radar 2- Last Position ##########################
#6.371,1.0582
Xt = Symbol('Xt')
Yt = Symbol('Yt')
d2_last = radar2[-1, 0]
phi2_last = radar2[-1, 1]
eq1 = sqrt(Xt ** 2 + Yt ** 2) - d2_last
# θ = 0 for initialization
eq2 = atan2(Yt, Xt) - phi2_last
robot_state = nsolve((eq1, eq2), (Xt, Yt), (-1, 1))
print(robot_state)
#[4.92849957419928], [5.62568436611205]]
