import numpy as np
import matplotlib.pyplot as plt
import random as r
from scipy.interpolate import splrep, splev
k=1e-12; kap=1e-1*3600; v0=1e-4*3600; p0=25.0; Myu=0.1*1e-8/3600;
lam_p = 1000 / 3600
L = 200.0
N = 401
M = 251
T = 2.0
d = int(input("d = "))
delta = float(input("delta = "))

SmoothingParam = float(input("Enter the smoothing parameter: "))

h = L / (N-1); tau = T / (M-1);
x = np.linspace(0, L, N)
#print("x = ", x)
t = np.linspace(0, T, M)
#print('t = ', t)
alpha = np.zeros(N)
betta = np.zeros(N)
ex_sol = np.zeros(M)
ap_sol = np.zeros(M)
p = np.zeros((M, N))

step = int(input("step = "))

p[0, :] = p0

for j in range(M-1):
    alpha[1] = 1;
    betta[1] = -v0 * Myu * h * tau / (k * (tau + lam_p)) - lam_p / (tau + lam_p) * (p[j, 1] - p[j, 0]);
    A = kap * (tau + lam_p) / h ** 2
    B = A
    C = 1 + 2 * A
    for i in range(1, N-1):
        F = p[j, i] - kap * lam_p * (p[j, i + 1] - 2 * p[j, i] + p[j, i - 1]) / h ** 2
        alpha[i+1] = B/(C-alpha[i]*A);
        betta[i+1] = (A * betta[i] + F) / (C - alpha[i] * A);
    p[j + 1, N - 1] = p0
    for i in range(N-2,-1,-1):
        p[j + 1, i] = alpha[i + 1] * p[j + 1, i + 1] + betta[i + 1];
for j in range(M):
    ex_sol[j] = p[j, 0];
#print("p = ",p)
print("Exact solution: ",list(ex_sol[0:M:step]))

z = np.zeros(len(t))
for j in range(0, M):
    z[j] = p[j, d] + 2 * delta * (np.random.rand() - 0.5)

print("z = ",list(z))

tck = splrep(range(M), z, s=SmoothingParam)

spline = splev(range(M), tck)

p2 = np.zeros((M, N))
alpha2 = np.zeros(N)
betta2 = np.zeros(N)

p2[0, :] = p0

for j in range(M-1):
    alpha2[d+1] = 0;
    betta2[d+1] = spline[j+1];
    A2 = kap * (tau + lam_p) / h ** 2
    B2 = A2
    C2 = 1 + 2 * A2
    for i in range(d+1, N-1):
        F2 = p2[j, i] - kap * lam_p * (p2[j, i + 1] - 2 * p2[j, i] + p2[j, i - 1]) / h ** 2
        alpha2[i+1] = B2 / (C2 - alpha2[i] * A2)
        betta2[i+1] = (A2 * betta2[i] + F2) / (C2 - alpha2[i] * A2)
    p2[j+1, N-1] = p0
    for i in range(N-2,d-1,-1):
        p2[j+1, i] = alpha2[i+1]*p2[j+1, i+1]+betta2[i+1]
    for i in range(d, 0, -1):
        F2 = p2[j, i] - kap * lam_p * (p2[j, i + 1] - 2 * p2[j, i] + p2[j, i - 1]) / h ** 2
        p2[j+1, i-1] = (C2 * p2[j+1, i] - B2 * p2[j+1, i+1] - F2) / A2
        #p2[j+1, i-1] = (2+(h**2)/(kap*(tau+lam_p)))*p2[j+1, i]-p2[j+1, i+1]-((h**2)*p2[j, i])/(kap*(tau+lam_p))+(kap*lam_p*(p2[j, i+1]-2*p2[j, i]+p2[j, i-1]))/(kap*(tau+lam_p))
#print("p2 = ", p2)
for j in range(M):
    ap_sol[j] = p2[j, 0]
print("Approximate solution: ", list(ap_sol[0:M:step]))

#subplot 1
sp = plt.subplot(221)
plt.plot(t, p[:, d], '--', linewidth=1)
plt.xlabel('t, hour')
plt.ylabel('z(t), MPa')
plt.grid()

#subplot 2
sp = plt.subplot(222)
for j in range(0, M, 50):
    plt.plot(x, p[j, :], '--', linewidth=1)
plt.xlabel('x')
plt.ylabel('P, MPa')
plt.grid()

#subplot 3
sp = plt.subplot(223)
plt.plot(t[0:M:step], ex_sol[0:M:step], label='Exact solution')
plt.plot(t[0:M:step], ap_sol[0:M:step], 'r--', label='Approximate solution')
plt.xlabel('t, hour')
plt.ylabel('p, MPa')
plt.legend(loc='best', fontsize=10)
plt.grid()

plt.show()
