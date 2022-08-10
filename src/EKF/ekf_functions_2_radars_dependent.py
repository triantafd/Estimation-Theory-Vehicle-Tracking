import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import *
import sympy
from sympy.abc import alpha, X, Y,  u, w, t, theta
from sympy import symbols, Matrix
from src.EKF.useful_functions import wrapToPi, plot_cov_ellipse
import time

def H_k(x, radar_pos):
    """ compute Jacobian of H matrix where h(x)

        #Find Jacobian H
        z = Matrix([[sympy.sqrt((px - x) ** 2 + (py - y) ** 2)],
                    [sympy.atan2(y - py, x - px) - theta]])
        print(z.jacobian(Matrix([x, y, theta])))
    """
    px = radar_pos[0]
    py = radar_pos[1]
    hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
    dist = sqrt(hyp)

    H = np.array(
        [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
         [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1]])
    return H


def H_x(x, radar_pos):#, radar_pos
    """ takes a state variable and returns the measurement
    that would correspond to that state.
    """
    px = radar_pos[0]
    py = radar_pos[1]
    dist = sqrt((px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2)

    #Hx = np.array([[dist],
    #               [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])

    Hx = np.array([[dist],
                   [atan2(x[1, 0] - py, x[0, 0] - px) - x[2, 0]]])
    return Hx


def ekf_predict(X_previous, P_previous, Q, dt, Uk, Fk, V):

    X_predict = fxu.evalf(subs={X: X_previous[0, 0], Y: X_previous[1, 0], u: Uk[0], w: Uk[1],
                        t: dt, theta: X_previous[2, 0]})

    X_previous[2, 0] = wrapToPi(X_previous[2, 0])

    Fk = Fk.evalf(subs={X: X_previous[0, 0], Y: X_previous[1, 0], u: Uk[0], w: Uk[1],
                        t: dt, theta: X_previous[2, 0]})

    #V = V.evalf(subs={X: X_previous[0, 0], Y: X_previous[1, 0], u: Uk[0], w: Uk[1],
    #                    t: dt, theta: X_previous[2, 0]})
    #Q = np.dot(V, np.dot(Q, V.T))

    P_predict = np.dot(Fk, np.dot(P_previous, Fk.T)) + Q

    return X_predict, P_predict

def ekf_update(X_predicted, P_predicted, zk1 ,  radar_pos1, zk2 , radar_pos2, Rk):

    zk1 = np.resize(zk1, (2, 1))
    zk2 = np.resize(zk2, (2, 1))
    print(zk1,zk2)
    zk = np.vstack((zk1, zk2))

    print('######################## (zk) observation is ##########################\n {}'.format(zk))
    Hx1 = H_x(X_predicted, radar_pos1)
    Hx1[1, 0] = wrapToPi(Hx1[1, 0])
    Hx2 = H_x(X_predicted, radar_pos2)
    Hx2[1, 0] = wrapToPi(Hx2[1, 0])
    Hx = np.vstack((Hx1, Hx2))
    print('######################## (Hx) Pre-fit residual	 ##########################\n {}'.format(Hx))
    Yk = zk - Hx
    Yk[1, 0] = wrapToPi(Yk[1, 0])
    Yk[3, 0] = wrapToPi(Yk[3, 0])
    print('######################## YK) Pre-fit residual	 ##########################\n {}'.format(Yk))
    Hk1 = H_k(X_predicted, radar_pos1)
    Hk2 = H_k(X_predicted, radar_pos2)
    Hk = np.vstack((Hk1, Hk2))
    print('######################## Hk observation_model ##########################\n {}'.format(Hk))
    Sk = np.dot(Hk, np.dot(P_predicted, Hk.T)) + Rk
    print('######################## (Sk) pre-fit residual covariance observation_model ##########################\n {}'.format(Sk))
    Kk = np.dot(P_predicted, np.dot(Hk.T, np.linalg.inv(Sk.astype(float))))
    print('######################## Optimal Kalman gain	 ##########################\n {}'.format(Kk))
    X_new = X_predicted + np.dot(Kk, Yk)
    X_new[2, 0] = wrapToPi(X_new[2, 0])
    print('######################## (Xk) Updated state estimate	 ##########################\n {}'.format(X_new))
    P_new = P_predicted - np.dot(Kk, np.dot(Hk, P_predicted))
    #P_new = P_predicted - np.dot(Kk, np.dot(Sk, Kk.T))
    #P_new = P_predicted - np.dot(P_predicted, np.dot(Hk.T, Kk.T))
    print('######################## (Pk) Updated estimate covariance	 ##########################\n {}'.format(P_new))

    return X_new, P_new

if __name__ == '__main__':
    radar1 = pd.read_csv('../../dataset/radar1.csv', header=None).to_numpy()
    radar2 = pd.read_csv('../../dataset/radar2.csv', header=None).to_numpy()
    control = pd.read_csv('../../dataset/control.csv', header=None).to_numpy()
    px, py, x, y, theta = symbols('p_x, p_y, x, y, theta')

    X_s = np.array([[10 - 6.3], [3.85], [0.785]])

    P = np.array([[0.001, 0, 0.0],
                  [0.0, 0.001, 0.0],
                  [0.0, 0.0, 0.001]])

    #define Q
    # Q = np.array([[0.0025, 0],
    #             [0, 0.01]])
    Q = np.array([[0.0025, 0.0, 0.0],
                  [0.0, 0.0025, 0.0],
                  [0.0, 0.0, 0.01]])

    fxu = Matrix([[X+ sympy.cos(theta) * u * t],
                  [Y + sympy.sin(theta) * u * t],
                  [theta + w * t]])

    Fk = fxu.jacobian(Matrix([X, Y, theta]))
    V = fxu.jacobian(Matrix([u, w]))

    #Define R
    Rk = np.array([[1, 0.0, 0.0, 0.0],
                    [0.0, 0.04, 0.0, 0.0],
                    [0.0, 0.0, 0.09, 0.0],
                    [0.0, 0.0, 0.0, 0.0025]])

    dt = 0.1
    correctedX1 = []
    correctedX2 = []


    start_time = time.time()
    for i in range(1, len(radar1)):

        X_s, P = ekf_predict(X_s, P, Q, dt, control[i], Fk, V)
        X_s, P = ekf_update(X_s, P, radar1[i],  [0, 0], radar2[i], [10, 0], Rk)
        plot_cov_ellipse(P[0:2, 0:2], [X_s[0,0], X_s[1,0]], 1)
        plt.scatter(X_s[0], X_s[1], color='b')

    end = time.time()
    print(end - start_time)

    plt.legend(["Corrected"])
    plt.ylabel('Y POSITION')
    plt.xlabel('X POSITION ')
    # plot title
    plt.title('2 RADARS')
    plt.show()

