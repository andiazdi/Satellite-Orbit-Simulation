import numpy as np
import matplotlib.pyplot as plt

def ode45(odefun, tspan, x0, **opts):
    t = tspan
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = odefun(t[i - 1], x[i - 1])
        k2 = odefun(t[i - 1] + dt / 2, x[i - 1] + dt / 2 * k1)
        k3 = odefun(t[i - 1] + dt / 2, x[i - 1] + dt / 2 * k2)
        k4 = odefun(t[i - 1] + dt, x[i - 1] + dt * k3)
        x[i] = x[i - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, x

def rotz(gamma):
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])

G = 6.673 * 10 ** (-11)
M = 5.972 * 10 ** 24
radius = 6371000
ISS_height = 437 * 10 ** 3
ISS_velocity = 6000
ISS_time = 90 * 60

Oz = np.array([0, 0, 1])

North = -43.07 * np.pi / 180
East = -61.50 * np.pi / 180

init_pos = np.array([np.cos(North) * np.cos(East),
                     np.cos(North) * np.sin(East),
                     np.sin(North)])

def get_orbit_n(r, solution_one):
    phi = 51.6 * np.pi / 180

    p1 = -r[1] / r[0]
    p2 = -np.cos(phi) * r[2] / r[0]

    a = p1 ** 2 + 1
    b = 2 * p1 * p2
    c = p2 ** 2 - np.sin(phi) ** 2

    y1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    y2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    x1 = p1 * y1 + p2
    x2 = p1 * y2 + p2

    z = np.cos(phi)

    n1 = np.array([x1, y1, z])
    n2 = np.array([x2, y2, z])

    if solution_one:
        orbit_norm = n1
    else:
        orbit_norm = n2

    return orbit_norm

orbit_norm = get_orbit_n(init_pos, False)
tau = np.cross(orbit_norm, init_pos)

r0 = init_pos * (radius + ISS_height)
v0 = tau * ISS_velocity

tspan = np.linspace(0, 1 * ISS_time, 10 ** 5)
x0 = np.concatenate((r0, v0))
odefun = lambda t, x: np.concatenate((x[3:6], -G * M * x[:3] / np.linalg.norm(x[:3]) ** 3))

opts = {'Reltol': 1e-13, 'AbsTol': 1e-14, 'Stats': 'on'}

t, x = ode45(odefun, tspan, x0, **opts)
trajectory = x[:, :3]

trajectory_corrected = np.zeros(trajectory.shape)
for i in range(len(t)):
    current_time = t[i]
    angle_Erth_rotation = -2 * np.pi * current_time / (24 * 60 * 60)
    current_point = trajectory[i, :]
    current_point_corrected = rotz(angle_Erth_rotation) @ current_point
    trajectory_corrected[i, :] = current_point_corrected

N = 100
phi = np.linspace(0, 2 * np.pi, N)
theta = np.linspace(0, np.pi, N)
theta, phi = np.meshgrid(theta, phi)

X = radius * np.cos(phi) * np.sin(theta)
Y = radius * np.sin(phi) * np.sin(theta)
Z = radius * np.cos(theta)

fig = plt.figure(figsize = [10, 10])
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, color='green', alpha=0.5)

ax.plot3D(trajectory_corrected[:, 0], trajectory_corrected[:, 1], trajectory_corrected[:, 2], linewidth=4)

ax.set_title('Satellite orbit')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig = plt.figure(figsize=(20, 10), dpi=80)

ax = fig.add_subplot(211) 
ax1 = fig.add_subplot(212)
ax.plot(tspan, x[:, 0:3])
ax.set_title('Coordinates')
ax1.plot(tspan, x[:, 3:6])
ax.set_title('Velocity')

plt.show()