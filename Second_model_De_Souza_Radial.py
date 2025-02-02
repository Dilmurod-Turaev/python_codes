import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
k = 1e-12; kap = 3e-1*3600; Q0 = 100 / 24; p0 = 25.0; Myu = 25e-9/3600
lam_p = 1000 / 3600
lam_v = 500 / 3600
R = 200
N = 401
M = 251
T = 2.0
H = 10
r_c = 0.1
d = int(input("d = "))
delta = float(input("delta = "))

SmoothingParam = float(input("Enter the smoothing parameter: "))

h = R / (N - 1)
tau = T / (M - 1)

r = np.linspace(0, R, N)
#print("r = ", r)

r1 = np.zeros(N)
r2 = np.zeros(N)

t = np.linspace(0, T, M)
#print('t = ', t)

alpha = np.zeros(N)
betta = np.zeros(N)
ex_sol = np.zeros(M)
ap_sol = np.zeros(M)
p = np.zeros((M, N))
A = np.zeros(N)
B = np.zeros(N)
C = np.zeros(N)
F = np.zeros(N)

step = int(input('step = '))

p[0, :] = p0
p[1, :] = p0

for i in range(N):
    r1[i] = i * h - 0.5 * h
    r2[i] = i * h + 0.5 * h
    A[i] = kap * r1[i] * (tau + lam_p) / h ** 2
    B[i] = kap * r2[i] * (tau + lam_p) / h ** 2
    C[i] = A[i] + B[i] + r[i] / 2 + (r[i] * lam_v) / tau

for j in range(1, M-1):
    alpha[1] = 1
    betta[1] = -Q0 * Myu * h / (2 * np.pi * H * k * r_c)
    for i in range(1, N-1):
        F = r[i] * (1 / 2 - lam_v / tau) * p[j-1, i] + ((2 * r[i] * lam_v) / tau + kap * lam_p * (r2[i] + r1[i]) / h ** 2) * p[j, i] - kap * lam_p * (r2[i] * p[j, i+1] + r1[i] * p[j, i-1]) / h ** 2
        alpha[i+1] = B[i] / (C[i] - alpha[i] * A[i])
        betta[i+1] = (A[i] * betta[i] + F) / (C[i] - alpha[i] * A[i])
    p[j+1, N-1] = p0
    i = N-2
    while i > -1:
        p[j+1, i] = alpha[i+1] * p[j+1, i+1] + betta[i+1]
        i = i - 1
for j in range(M):
    ex_sol[j] = p[j, 0]
#print("p = ",p)
ex_sol = [float(x) for x in ex_sol]
print("Exact solution: ", ex_sol[0:M:step])

z = np.zeros(len(t))
for j in range(0, M):
    z[j] = p[j, d] + 2 * delta * (np.random.rand() - 0.5)
z = [float(x) for x in z]
print("z = ", z)

tck = splrep(range(M), z, s = SmoothingParam)
spline = splev(range(M), tck)

p2 = np.zeros((M, N))
alpha2 = np.zeros(N)
betta2 = np.zeros(N)

p2[0, :] = p0
p2[1, :] = p0

for j in range(1, M-1):
    alpha2[d+1] = 0
    betta2[d+1] = spline[j+1]
    for i in range(d+1, N-1):
        F2 = r[i] * (1 / 2 - lam_v / tau) * p2[j-1, i] + ((2 * r[i] * lam_v) / tau + kap * lam_p * (r2[i] + r1[i]) / h ** 2) * p2[j, i] - kap * lam_p * (r2[i] * p2[j, i+1] + r1[i] * p2[j, i-1]) / h ** 2
        alpha2[i+1] = B[i] / (C[i] - alpha2[i] * A[i])
        betta2[i+1] = (A[i] * betta2[i] + F2) / (C[i] - alpha2[i] * A[i])
    p2[j+1, N-1] = p0
    for i in range(N-2,d-1,-1):
        p2[j+1, i] = alpha2[i+1] * p2[j+1, i+1] + betta2[i+1]
    for i in range(d, 0, -1):
        F2 = r[i] * (1 / 2 - lam_v / tau) * p2[j-1, i] + ((2 * r[i] * lam_v) / tau + kap * lam_p * (r2[i] + r1[i]) / h ** 2) * p2[j, i] - kap * lam_p * (r2[i] * p2[j, i+1] + r1[i] * p2[j, i-1]) / h ** 2
        p2[j+1, i-1] = (C[i] * p2[j+1, i] - B[i] * p2[j+1, i+1] - F2) / A[i]
#print("p2 = ", p2)

for j in range(M):
    ap_sol[j] = p2[j, 0]
ap_sol = [float(x) for x in ap_sol]
print("Approximate solution: ", ap_sol[0:M:step])

#subplot 1
plt.subplot(221)
plt.plot(t, z, linewidth=1)
plt.xlabel('t, hour')
plt.ylabel('z(t), MPa')
plt.grid()

#subplot 2
plt.subplot(222)
for j in range(0, M, 50):
    plt.plot(r, p[j, :], '--', linewidth=1)
plt.xlabel('r')
plt.ylabel('P, MPa')
plt.grid()

#subplot 3
plt.subplot(223)
plt.plot(t[0:M:step], ex_sol[0:M:step], label='Exact solution')
plt.plot(t[0:M:step], ap_sol[0:M:step], 'r--', label='Approximate solution')
plt.xlabel('t, hour')
plt.ylabel('p, MPa')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.show()
