import numpy as np

import matplotlib.pyplot as plt


def optimise_lorentzian(x, y, maxiter=100_000, eps=1e-15, update_strength=0.0001):
	#starting parameters
	y_half = (y.max() - y.min())/2
	y_idx_closest_to_half = np.argsort((y-y_half)**2)[:4]
	w = float(np.diff(x[y_idx_closest_to_half]).max())
	P0 = x.mean()
	B = y.min()

	#functions
	L = lambda x, w, P0, B: (1+((x-P0)/(w/2))**2)**-1 + B
	error = lambda x, w, P0, B: np.sum((y-L(x, w, P0, B))**2)

	errors = []
	for i in range(maxiter):
		errors.append(error(x, w, P0, B))
		if i > 10 and all([abs(e - errors[-1]) < eps for e in errors[-8:]]): break #final conditions

		#get derivatives
		lorentzian = L(x, w, P0, B)
		dw = -16 * np.sum( (y-lorentzian)*lorentzian**2 * (x-P0)**2/(w**3) )
		dP0 = -16 * np.sum( (y-lorentzian)*lorentzian**2 * (x-P0)/(w**2) )
		dB = -2 * np.sum( y-lorentzian )

		#update parameters
		w  = w  - update_strength * dw
		P0 = P0 - update_strength * dP0
		dB = B  - update_strength * dB


	return w, P0, B, np.asarray(errors)




L = lambda x, w, P0, B: (1+((x-P0)/(w/2))**2)**-1 + B

x = np.linspace(-1,2,1000)
y = L(x, 0.5, 0.5, 0.1) + np.random.random(x.shape)/5

w, P0, B, errors = optimise_lorentzian(x, y)
predict_y = L(x, w, P0, B)
print(f'w  = {w}')
print(f'P0 = {P0}')
print(f'B  = {B}')
plt.scatter(x, y, label='y')
plt.plot(x, predict_y, label='y_pred', color='red')


plt.legend()
plt.show()

plt.plot(range(len(errors)), np.log(errors))
plt.show()