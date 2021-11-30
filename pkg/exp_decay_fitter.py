import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt




def fitter(f, x, y, vars=[], name='fit', p0=None, show_fit=False, plots_dir=None):
	error = lambda x, *args: np.sum((y-f(x, *args))**2)
	res = curve_fit(f, x, y, maxfev=100_000)[0]
	Stot = np.sum((y-np.mean(y))**2)
	r2_value = 1 - (error(x, *res)/Stot)

	results = {v:r for v, r in zip(vars, res.values())}
	

	# if show_fit:






def fit_exp_lin(x, y, show_fit=False, plots_dir=None, title='Predicted and reference rate', index=0, show_residuals=False):
	f = lambda x, k, A0, b: A0 * np.exp(-k*x) + b * x
	error = lambda x, *args: np.sum((y-f(x, *args))**2)

	halfheight = (y.max()-y.min())/2 + y.min()
	closest_to_middle = np.argmin(np.abs(y-halfheight))
	k = np.log(2)/x[closest_to_middle]
	A0 = (y.max()-y.min())

	res = curve_fit(f, x, y, [k, A0, 0], maxfev=10000)[0]

	Stot = np.sum((y-np.mean(y))**2)
	r2_value = 1 - (error(x, *res)/Stot)

	k = res[0]
	A0 = res[1]
	b = res[2]
	results = {'k':k, 'A0':A0, 'b':b, 'error':error(x, *res), 'r2_value':r2_value}

	if show_fit: 
		plt.figure()
		plt.plot(x, f(x, *res), label=f'Predicted decay, R2={r2_value}')
		plt.plot(x, y, label='Reference data')
		plt.title(f'{title} {index}\nk={k:.3f}, A0={A0:.3f}, b={b:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('Transmittance')
		plt.legend()
		plt.tight_layout()
		if plots_dir is not None: plt.savefig(f'{plots_dir}/exp_decay_lin_fit_{index}.jpg')

	if show_residuals: 
		plt.figure()
		plt.plot(x, y-f(x, *res))
		plt.plot(x, np.zeros_like(x), linestyle='dashed', color='black')
		plt.title(f'{title} residuals {index}\nk={k:.3f}, A0={A0:.3f}, b={b:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('y - f(x|k,A0,b)')
		plt.tight_layout()
		if plots_dir is not None: plt.savefig(f'{plots_dir}/exp_decay_lin_residuals_{index}.jpg')

	plt.figure()
	plt.plot(x, y)
	plt.plot(x, y - f(x, *res))
	plt.show()


	return results


def fit_exp_decay(x, y, use_scipy=True, maxiter=100_000, eps=1e-12, plot_errors=False, show_fit=True, 
				  plots_dir=None, title='Predicted and reference rate', outfile=None, index=0, show_residuals=True):
	#x is usually time, y is usually tracked wavelength
	#fitting to [A]_0 exp(-kt) + B
	#initial values
	# k = 2/x.max()			#height of 0.1 + B at time x.max()/2
	halfheight = (y.max()-y.min())/2 + y.min()
	closest_to_middle = np.argmin(np.abs(y-halfheight))
	k = np.log(2)/x[closest_to_middle]
	A0 = (y.max()-y.min())			

	f = lambda x, k, A0: A0 * np.exp(-k*x) 
	error = lambda x, k, A0: np.sum((y-f(x, A0, k))**2)

	res = curve_fit(f, x, y, [k, A0], maxfev=10000)[0]
	k = res[0]
	A0 = res[1]

	Stot = np.sum((y-np.mean(y))**2)
	r2_value = 1 - (error(x, *res)/Stot)
	results = {'model': 'exp_decay', 'k':k, 'A0':A0, 'error':error(x, *res), 'r2_value':r2_value}


	if show_fit: 
		plt.figure()
		plt.plot(x, f(x,*res), label=f'Predicted decay, R2={r2_value}')
		plt.plot(x, y, label='Reference data')
		plt.title(f'{title} {index}\nt={1/k:.2f}, A0={A0:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('Transmittance')
		plt.hlines(A0/2, x.min(), np.log(2)/k, colors=['black'], linewidths=[.75])
		plt.vlines(np.log(2)/k, y.min(), A0/2, colors=['black'], linewidths=[.75])
		plt.scatter(np.log(2)/k, A0/2, c='black')
		plt.text(np.log(2.2)/k, A0/1.85, r'$t_{1/2} = $' + f'{np.log(2)/k:.2f} s')
		plt.legend()
		plt.tight_layout()
		if plots_dir is not None: plt.savefig(f'{plots_dir}/exp_decay_fit_{index}.jpg')

	if show_residuals: 
		plt.figure()
		plt.plot(x, y-f(x, *res))
		plt.plot(x, np.zeros_like(x), linestyle='dashed', color='black')
		plt.title(f'{title} residuals {index}\nt={1/k:.2f}s, A0={A0:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('y - f(x|k,A0)')
		plt.tight_layout()
		if plots_dir is not None: plt.savefig(f'{plots_dir}/exp_decay_residuals_{index}.jpg')

	plt.figure()
	plt.plot(x, y)
	plt.plot(x, y - f(x, k, A0))
	plt.show()
	return results


def fit_biexp_decay(x, y, p0=None, show_fit=True, 
				  plots_dir=None, title='Predicted and reference rate', outfile=None, index=0, show_residuals=True):
	#x is usually time, y is usually tracked wavelength
	#fitting to [A]_0 exp(-kt) + B
	#initial values
	# k = 2/x.max()			#height of 0.1 + B at time x.max()/2
	if p0 is None:
		halfheight = (y.max()-y.min())/2 + y.min()
		closest_to_middle = np.argmin(np.abs(y-halfheight))
		k1 = np.log(2)/x[closest_to_middle]
		A01 = (y.max()-y.min())
		k1 = 1/200
		A01 = .8
		k2 = 1/10000
		A02 = .2

		p0 = [k1, k2, A01, A02, 0]

	f = lambda x, k1, k2, A01, A02, B: A01 * np.exp(-k1*x) + A02 * np.exp(-k2*x) + B
	error = lambda x, *args: np.sum((y-f(x, *args))**2)

	res = curve_fit(f, x, y, p0, bounds=[[0,0,0,0,-np.inf], [np.inf,np.inf,np.inf,np.inf,np.inf]], maxfev=10000)[0]
	
	k1 = res[0]
	k2 = res[1]
	A01 = res[2]
	A02 = res[3]
	B = res[4]

	Stot = np.sum((y-np.mean(y))**2)
	r2_value = 1 - (error(x, *res)/Stot)
	results = {'model': 'biexp_decay', 'k1':k1, 'k2':k2, 'A01':A01, 'A02':A02, 'B':B, 'error':error(x, *res), 'r2_value':r2_value}


	if show_fit: 
		plt.figure()
		plt.plot(x, f(x, *res), label=f'Predicted decay tot, R2={r2_value}', linewidth=2)
		plt.plot(x, A01*np.exp(-k1*x) + B, label=f'Predicted decay 1')
		plt.plot(x, A02*np.exp(-k2*x) + B, label=f'Predicted decay 2')
		plt.scatter(x, y, label='Reference data')
		plt.title(f'{title} {index}\nt1={1/k1:.1f}, t2={1/k2:.1f}, A01={A01:.3f}, A02={A02:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('Transmittance')
		plt.legend()
		plt.tight_layout()
		if plots_dir is not None: plt.savefig(f'{plots_dir}/biexp_decay_fit_{index}.jpg')

	if show_residuals: 
		plt.figure()
		plt.plot(x, y-f(x, *res))
		plt.plot(x, np.zeros_like(x), linestyle='dashed', color='black')
		plt.title(f'{title} residuals {index}\nt1={1/k1:.1f}, t2={1/k2:.1f}, A01={A01:.3f}, A02={A02:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('y - f(x|k1,k2,A01,A02)')
		plt.tight_layout()
		if plots_dir is not None: plt.savefig(f'{plots_dir}/biexp_decay_residuals_{index}.jpg')

	plt.figure()
	plt.plot(x, y)
	plt.plot(x, y - f(x, *res))
	# plt.show()
	return results



if __name__ == '__main__':
	import matplotlib.pyplot as plt

	f = lambda x, k, A0, B: A0 * np.exp(-k*x) + B


	x = np.linspace(-1,1,1000)
	y = f(x, 1, 0.9, 0.1) + np.random.random(x.shape)/100


	y = np.array([0.15349171, 0.15339416, 0.15351357, 0.1533864,  0.15354462, 0.153518,
 0.15346363, 0.15341833, 0.15347431, 0.15341405, 0.15354731, 0.15352348,
 0.15345234, 0.1535392 , 0.153526  , 0.15349127, 0.15351175, 0.15349331,
 0.15346805, 0.15344647, 0.15344195, 0.15350094, 0.15349855, 0.15355208,
 0.15339782, 0.15351598, 0.15342625, 0.15334652, 0.15348612, 0.15349408,
 0.15339576, 0.15339054, 0.15343784, 0.15334199, 0.15341767, 0.15344519,
 0.15335192, 0.15345505, 0.15347037, 0.15348859, 0.15336128, 0.15345922,
 0.15338691, 0.15335967, 0.15338669, 0.15343726, 0.15331631, 0.15335974,
 0.15336118, 0.15332398, 0.15334098, 0.15334466, 0.15336297, 0.15328216,
 0.15325064, 0.15328163, 0.15334921, 0.15334129, 0.15324712, 0.15338901,
 0.15330613, 0.15331152, 0.15341912, 0.1532401 , 0.15324256, 0.15336303,
 0.1533832 , 0.15329199, 0.15329328, 0.15329151, 0.15314379, 0.15326374,
 0.15326847, 0.15338853,])
	x = np.arange(y.size)

	opt_res = fit_exp_decay(x, y, show_fit=False, plot_errors=False)
	predict_y = f(x, opt_res["k"], opt_res["A0"], opt_res["B"])

	print('optimization complete:')
	print(f'k  = {opt_res["k"]}')
	print(f'A0 = {opt_res["A0"]}')
	print(f'B  = {opt_res["B"]}')

	plt.plot(x, y, label='y')
	plt.plot(x, predict_y, label='y_pred', color='red')


	plt.legend()
	plt.show()