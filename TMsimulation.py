import numpy as np
import matplotlib.pyplot as plt 


rH = np.sqrt(0.99)
tH = np.sqrt(1-rH*rH)
ng = 3.5
nP = 3.5
Lp = 300
Lg = .3
d = 4
FSR = 150
fd = np.pi * d/FSR
np1 = 1
Lp1 = 3
r1 = r2 = np.sqrt(0.539)
t1 = np.sqrt(1-r1*r1)
t2 = np.sqrt(1-r2*r2)
ne = 1.5
Le = 0.025
np2 = 1
Lp2 = 1
rM = np.sqrt(0.9)
tM = np.sqrt(1-rM*rM)
g = 1/(2*Lg) * np.log((1-fd)/(rH*rM))


k = np.linspace(400, 7000, 100000)

H = np.array([[1/tH, rH/tH],[rH/tH, 1/tH]])
G = np.array([[np.exp(g*Lg-k*ng*Lg*1j), 0], [0, np.exp(-g*Lg+k*ng*Lg*1j)]])
P = np.array([[np.exp(-k*nP*Lp*1j),0], [0, np.exp(k*nP*Lp*1j)]])
P1 = np.array([[np.exp(-k*np1*Lp1*1j),0], [0, np.exp(k*np1*Lp1*1j)]])
P2 = np.array([[np.exp(-k*np2*Lp2*1j),0], [0, np.exp(k*np2*Lp2*1j)]])
E = np.array([[t1*t2*np.exp(-k*ne*Le*1j)/(1-r1*r2*np.exp(-2*k*ne*Le*1j)), 0], [0, (1-r1*r2*np.exp(-2*k*ne*Le*1j))/(t1*t2*np.exp(-k*ne*Le*1j))]])
M = np.array([[1/tM, rM/tM], [rM/tM, 1/tM]])

R = M @ P2 @ E @ P1 @ P @ G
Efp = 1
ED = -rH * R[1,0]/R[1,1] * Efp
Einf = Efp/(1+rH*R[1,0]/R[1,1])
Eout = Einf*(R[0,0]-R[0,1]*R[1,0]/R[1,1])

Eout2 = Eout * np.conjugate(Eout)
print(Eout2.astype(float))
plt.plot(k, Eout2)
plt.show()