import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import uniform
from numpy.random import rand
from scipy.stats import multivariate_normal
from scipy.stats import norm
from datetime import datetime
from filterpy.monte_carlo import systematic_resample
import os
import time


def create_uniform_particles(x_range, y_range, theta_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(theta_range[0], theta_range[1], size=N)
    for i in range(0, N):
        particles[i, 2] = wrapToPi(particles[i, 2])
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    for i in range(0, N):
        particles[i, 2] = wrapToPi(particles[i, 2])
    return particles


# Wrap radians to [−pi pi]
def wrapToPi(d):
    # Wrap to [0..2*pi]
    d = d % (2 * np.pi)
    # Wrap to [-pi..pi]
    if d > np.pi:
        d -= (2 * np.pi)
    return d


# Wrap radians to [−pi pi]
def wrapToPiN(d, N):
    # Wrap to [0..2*pi]
    for i in range(0, N):
        d[i] = d[i] % (2 * np.pi)
        if d[i] > np.pi:
            d[i] -= 2 * np.pi
    return d


def neff(weights):
    return 1. / np.sum(np.square(weights))


class ParticleFilter:

    def __init__(self, N,  Q, robot, dt):
        # N : Number of particles
        # landmarks : Landmarks positions
        # particles : Particles
        # weights : Weights
        # Q : Noise std
        # R : Noise

        self.dt = dt
        self.N = N
        self.robot = robot
        self.weights = np.ones(N) / N
        self.Q = Q
        # Randomly generate N particles
        #self.particles = create_uniform_particles((0, 5), (0, 9), (-3.14 , 3.14), N)
        self.particles = create_gaussian_particles(mean=robot, std=self.Q, N=N)

    def predict(self, u, dt):

        N = self.N
        std = self.Q
        # Move in the (noisy) commanded direction
        self.particles[:, 0] = self.particles[:, 0] + np.cos(self.particles[:, 2]) * (u[0] + (randn(N) * std[0])) * dt
        self.particles[:, 1] = self.particles[:, 1] + np.sin(self.particles[:, 2]) * (u[0] + (randn(N) * std[1])) * dt
        self.particles[:, 2] = self.particles[:, 2] + dt * (u[1] + randn(N) * std[2])
        for i in range(0, N):
            self.particles[i, 2] = wrapToPi(self.particles[i, 2])

    def update(self, z, radar, R):
        print(self.weights)
        print(radar)
        print(self.particles)
        distance = np.linalg.norm(self.particles[:, 0:2] - radar, axis=1)
        angle = np.array(wrapToPiN(np.arctan2(self.particles[:, 1] - radar[1],
                                              self.particles[:, 0] - radar[0]) - self.particles[:, 2], self.N))
        #angle = np.array(wrapToPiN(np.arctan2(radar[1] - self.particles[:, 1],
         #                                    radar[0] - self.particles[:, 0]) - self.particles[:, 2], self.N))

        distance = np.array((np.array([distance, angle])).T)

        self.weights *= [normpdf(x=z[0], mu=distance[j], std=R) for j in range(0, np.size(distance, 0))]

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # Normalize to sum to 1

    def estimate(self):
        robotPosition = self.particles[:, 0:3]
        mean = np.average(robotPosition, weights=self.weights, axis=0)
        var = np.average((robotPosition - mean) ** 2, weights=self.weights, axis=0)
        return mean, var

    def multinomial_resample(self):
        N = self.N
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, rand(len(self.weights)))

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / N)

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))

    def residual_resample(self):
        N = self.N
        indexes = np.zeros(self.N, 'i')

        # take int(N*w) copies of each weight
        num_copies = (N * np.asarray(self.weights)).astype(int)
        k = 0
        for i in range(N):
            for _ in range(num_copies[i]):  # make n copies
                indexes[k] = i
                k += 1

        # use multinormial resample on the residual to fill up the rest.
        residual = self.weights - num_copies  # get fractional part
        residual /= sum(residual)  # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.  # ensures sum is exactly one
        indexes[k:N] = np.searchsorted(cumulative_sum, rand(N - k))

        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))

    def stratified_resample(self):
        N = self.N
        # make N subdivisions, chose a random position within each one
        print(rand(N))
        print(range(N))
        positions = (rand(N) + range(N)) / N

        indexes = np.zeros(N, 'i')
        print(self.weights)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))

    def systematic_resample(self):
        N = self.N

        # make N subdivisions, choose positions
        # with a consistent random offset
        positions = (np.arange(N) + rand()) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))


def normpdf(x, mu, std):
    cov = np.diag((1, 1)) * std
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))


def mainFunction():
    # Read Datasets
    #Q = np.array([[pow(0.05, 2), 0], [0, pow(0.1, 2)]])
    Q = np.array([0.05, 0.05, 0.1])
    R1 = np.array([[1, 0],
                   [0, 0.04]])
    R2 = np.array([[0.09, 0],
                   [0, 0.0025]])
    radar1 = pd.read_csv('../../dataset/radar1.csv', header=None).to_numpy()
    radar2 = pd.read_csv('../../dataset/radar2.csv', header=None).to_numpy()
    control = pd.read_csv('../../dataset/control.csv', header=None).to_numpy()

    # Convert radians to [-pi, pi]
    for i in range(0, len(radar1)):
        radar1[i][1] = wrapToPi(radar1[i][1])
        radar2[i][1] = wrapToPi(radar2[i][1])

    '''
    # Values I want to find by solving the measurement model
    Xt = Symbol('Xt')
    Yt = Symbol('Yt')
    X_s = np.array([[6.3], [3.85], [0.0]]).astype(float)
    d2 = radar2[0, 0]
    phi2 = radar2[0, 1]
    eq1 = sqrt(Xt ** 2 + Yt ** 2) - d2
    # θ = 0 for initialization
    eq2 = atan2(Yt, Xt) - phi2
    robot_state = nsolve((eq1, eq2), (Xt, Yt), (-1, 1))
    
    
    X_s = np.array([[4.58], [1.36], [0.0]]).astype(float)
    d1 = radar1[0, 0]
    phi1 = radar1[0, 1]
    eq1 = sqrt(Xt ** 2 + Yt ** 2) - d1
    # θ = 0 for initialization
    eq2 = atan2(Yt, Xt) - phi1
    robot_state = nsolve((eq1, eq2), (Xt, Yt), (-1, 1))
    '''

    # First position of robot
    robot = np.array([4.58, 1.36, 0])
    #robot = np.array([10 - 6.3, 3.85, 0])
    # Number of particles
    N = 100
    dt = 0.1

    radars = np.array([[0, 0], [10, 0]])
    pf = ParticleFilter(N=N, Q=Q, robot=robot, dt=dt)

    resampling = 0
    resampleIndex = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlim(-2, 10)
    ax.set_ylim(-2, 9)
    ax.scatter(radars[1][0], radars[1][1], color='green')
    ax.scatter(radars[0][0], radars[0][1], color='black')
    for i in range(1, len(radar1)):
        ut = control[i, :]
        z1 = np.array([radar1[i]])
        #z2 = np.array([radar2[i]])
        pf.predict(ut, dt)
        pf.update(z1, [0,0], R1)
        #pf.update(z2, [10,0], R2)

        if neff(pf.weights) < N * 0.75:
            # Multinomial resample
            # pf.multinomial_resample()
            # Resample from index
            #indexes = systematic_resample(pf.weights)
            #pf.resample_from_index(indexes)
            # Residual Resampling
            #pf.residual_resample()
            # Stratified Resampling
            pf.stratified_resample()
            # Systematic Resampling
            # pf.systematic_resample()
            resampling = resampling + 1
            resampleIndex.append(i)

        mean, var = pf.estimate()

        ax.set_title('Particle Filter Radar1 - iteration=%d' % (i + 1))
        p1 = ax.scatter(pf.particles[:, 0], pf.particles[:, 1], color='r', marker=',', s=1)
        p2 = ax.scatter(mean[0], mean[1], color='b', marker=',', s=1)
        if (i == 0):  # gia to capture tou video
            plt.pause(1)
        plt.pause(0.01)
        #filename = "plot" + str(i) + ".png"
        #fig.savefig(os.path.join("gif/", filename), bbox_inches='tight')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.legend([p1, p2], ['Particles', 'Estimation'], loc=4, numpoints=1)
        # sleep(0.5)


    print('Final position error, variance:\n\t', mean, var)
    print("Resampling: ", resampling)
    print("Indexes of resampling: ", resampleIndex)

np.random.seed(2020)
mainFunction()