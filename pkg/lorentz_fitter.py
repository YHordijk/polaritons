import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fit_lorentzian(x, y):
	#starting parameters
	y_half = (y.max() - y.min())/2
	y_idx_closest_to_half = np.argsort((y-y_half)**2)[:4]
	w = abs(float(np.diff(x[y_idx_closest_to_half]).max()))
	A = y.max() - y.min()
	P0 = x[y.argmax()]
	B = y.min()

	#functions
	L = lambda x, w, A, P0, B: A*(1+((x-P0)/(w/2))**2)**-1 + B
	error = lambda x, w, A, P0, B: np.sum((y-L(x, w, A, P0, B))**2)

	try:
		res = curve_fit(L, x, y, [w, A, P0, B], bounds=([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))[0]
	except:
		raise
	
	w, A, P0, B = res[0], res[1], res[2], res[3]
	return {'w':res[0], 'A':res[1], 'P0':res[2], 'B':res[3], 'error':error(x, w, A, P0, B), 'ymax':L(P0, w, A, P0, B)}



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	L = lambda x, w, A, P0, B: A * (1+((x-P0)/(w/2))**2)**-1 + B

	x = np.linspace(-1,2,1000)
	y = L(x, 1, 3, -0.9, 0.1) + np.random.random(x.shape)/10

	opt_res = fit_lorentzian(x, y)
	predict_y = L(x, opt_res["w"], opt_res["A"],  opt_res["P0"], opt_res["B"])

	print('optimization complete:')
	print(f'w  = {opt_res["w"]}')
	print(f'P0 = {opt_res["P0"]}')
	print(f'B  = {opt_res["B"]}')
	print(f'A  = {opt_res["A"]}')

	plt.scatter(x, y, label='y')
	plt.plot(x, predict_y, label='y_pred', color='red')


	plt.legend()
	plt.show()
