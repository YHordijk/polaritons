import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def read_UV(file):
	data = np.genfromtxt(file, skip_header=1, delimiter=',')[:,:-1]
	return data


def fit_exp_decay(x, y, maxiter=10_000, eps=1e-6, plot_errors=True, show_fit=True):
	#fitting to [A]_0 exp(-kt) + B
	#initial values
	k = 2/x.max()			#height of 0.1 + B at time x.max()/2
	A0 = (y.max()-y.min()) 	
	B = y.min() 			

	f = lambda x, A0, k, B: A0 * np.exp(-k*x) + B
	error = lambda x, A0, k, B: np.sum((y-f(x, A0, k, B))**2)

	# plt.plot(x, y)
	# cmap = plt.get_cmap('hot')
	errors = []
	for i in range(maxiter):
		errors.append(error(x, A0, k, B))
		# if errors[-1] < eps: break #check for completion of minimization
		if i > 10 and all([(e - errors[-1]) < eps for e in errors[-8:]]): break

		#calculate derivatives
		expdec = np.exp(-k*x)
		base = A0*expdec + B - y
		dA0 = np.sum(2*expdec			* base)
		dk  = np.sum(2*x*A0*expdec*-1	* base)
		dB  = np.sum(2			 		* base)

		#change parameters
		A0 = A0 - dA0 * 0.00001
		k = k - dk * 0.00001
		B = B - dB * 0.00001

	if i < maxiter:
		print(f'Exponential decay fit converged in {len(errors)}/{maxiter} iterations.')
	else:
		print(f'Exponential decay fit not converged. Will take more than {maxiter} iterations, or is not possible. ')

	if plot_errors: 
		plt.figure()
		plt.plot(range(len(errors)), errors)
		plt.title('Error during fitting')
		plt.xlabel('Iteration')
		plt.ylabel('Error (a.u.)')

	if show_fit: 
		plt.figure()
		plt.plot(x, f(x, A0, k, B), label='Predicted decay')
		plt.plot(x, y, label='Reference data')
		plt.title(f'Predicted and reference exponential decay\nA0={A0:.3f}, k={k:.3f}, B={B:.3f}')
		plt.xlabel('Time (s)')
		plt.ylabel('Absorbance (a.u.)')
		plt.hlines(B + A0/2, x.min(), np.log(2)/k, colors=['black'], linewidths=[.75])
		plt.vlines(np.log(2)/k, y.min(), B + A0/2, colors=['black'], linewidths=[.75])
		plt.scatter(np.log(2)/k, B + A0/2, c='black')
		plt.text(np.log(2.2)/k, B + A0/1.85, r'$t_{1/2} = $' + f'{np.log(2)/k:.2f} s')
		plt.legend()

	print('Parameters:')
	print(f'\tA0 = {A0} (a.u.)')
	print(f'\tk  = {k} (1/s)')
	print(f'\tB  = {B} (a.u.)')

	print(f'Mean Lifetime (1/k) \tτ    = {1/k:.2f} s')
	print(f'Half-life (τln(2))  \tt₁/₂ = {np.log(2)/k:.2f} s')

	return A0, k, B




if __name__ == '__main__':
	### KINETICS

	# data = read_UV('data/20211102 kinetic nsp in cavity/kinetic_data_20sinterval_no_uv.csv')
	data = read_UV('data/20211102 kinetic nsp in cavity/kinetic_data_10sinterval.csv')
	time_delay = 20
	tracking_wavelengths = [224, 340, 500]
	tracking_wavelengths = [550]
	tracking_wavelengths_k = 20 #wavelengths around tracking_wavelenghts to average around

	wavelenghts = data[:,0]
	absorbances = data[:,1:]
	cmap = plt.get_cmap('jet')
	N_spectra = absorbances.shape[1]

	print(f'Found {N_spectra} spectra! λ ⊂ [{wavelenghts[0]} nm, {wavelenghts[-1]} nm]')

	plt.figure()
	plt.title('Spectra')
	plt.xlabel('λ (nm)')
	plt.ylabel('Absorbance (a.u.)')
	for i, absorbance in enumerate(absorbances.T):
		plt.plot(wavelenghts, absorbance , c=cmap(i/N_spectra))

	
	def get_tracking_absorbance(wavelenght, k=5):
		if type(wavelenght) is tuple:
			low = wavelenght[0]
			high = wavelenght[1]
		else:
			low = wavelenght - k 
			high = wavelenght + k

		prev_wl = 0
		for i, wl in enumerate(wavelenghts):
			if prev_wl <= low and wl >= low:
				if k == 0:
					return absorbances[i-1] + (absorbances[i] - absorbances[i-1]) * (low-prev_wl)/(wl-prev_wl)
				else:
					low_i = i-1
			elif prev_wl <= high and wl >= high:
				high_i = i-1

			prev_wl = wl

		return np.mean(absorbances[low_i:high_i,:], axis=0)


	tracking_absorbances = []
	for twl in tracking_wavelengths:
		absorbs = get_tracking_absorbance(twl, tracking_wavelengths_k)
		tracking_absorbances.append(absorbs)

		plt.scatter(np.ones_like(absorbs) * twl, absorbs)

	plt.figure()
	plt.title('Tracked wavelenghts')
	plt.xlabel('t (s)')
	plt.ylabel('Absorbance (a.u.)')
	for twl, tabs in zip(tracking_wavelengths, tracking_absorbances):
		if type(twl) is tuple:
			low = twl[0]
			high = twl[1]
		else:
			low = twl - tracking_wavelengths_k
			high = twl + tracking_wavelengths_k
		
		plt.plot(np.arange(N_spectra)*time_delay, tabs, label=r'$\lambda_{avg}$' + f' = [{low}, {high}] nm')

	plt.legend()


	for twl, tabs in zip(tracking_wavelengths, tracking_absorbances):
		fit_exp_decay(np.arange(N_spectra-6)*time_delay, tabs[6:])


	plt.show()