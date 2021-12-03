import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



use_biexp = False


data = np.genfromtxt("kinetics_sp_cyclohexanone_1.txt", skip_header=1)
t = data[:,1]
A = data[:,2]


#subtract baseline
A = A - A[-20:-1].mean()


if not use_biexp:
	f = lambda x, A0, tau: A0 * np.exp(-x/tau)
	pinit = [200, 90]
else:
	f = lambda x, A01, A02, tau1, tau2: A01*np.exp(-x/tau1) + A02*np.exp(-x/tau2)
	pinit = [200, 0, 90, 4000]


res, _ = curve_fit(f, t, A, pinit)


if not use_biexp:
	print('Results:')
	print(f'\tt = {res[1]:.3f} s')

else:
	print('Results:')
	print(f'\tt1 = {res[2]:.3f} s')
	print(f'\tt2 = {res[3]:.3f} s')



plt.plot(t,A)
plt.plot(t,f(t, *res))
plt.show()


