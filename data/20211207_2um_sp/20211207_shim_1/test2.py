import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


f = lambda x, A0, t, a: A0 * np.exp(-(x-a)/t)

x = np.linspace(0, 10, 100)

def fit(x, *args):
	f2 = lambda x, *args: args[0]*np.exp(-x/args[1])
	return (*curve_fit(f2, x, f(x, *args), (1, 1))[0], 0)


# plt.plot(x, f(x,         1,1,0))
# plt.plot(x, f(x, *fit(x, 1,1,0)))

plt.plot(x, f(x,         1,1,-10), label='exp')
plt.plot(x, f(x, *fit(x, 1,1,-10)), label='fit')
plt.legend()

plt.show()