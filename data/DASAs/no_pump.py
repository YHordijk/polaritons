import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks 
from scipy.optimize import curve_fit


def file_reader(file, skip_header=1):
	with open(file, 'r') as f:
		content = f.readlines()

		if any(';' in line for line in content):
			content = [line.replace(',', '.').replace(';', ',') for line in content]

	return np.genfromtxt(content, skip_header=skip_header, delimiter=',')[:,:-1]


def get_average_signal_between(x, y, low, high):
	lowi = get_wavelenght_index(x, low)
	highi = get_wavelenght_index(x, high)

	avg = np.mean(y[lowi:highi+1,:], axis=0)
	return avg


def get_wavelenght_index(x, l):
	index = np.argsort(np.abs(x-l))[0]
	return index


def get_minima(x, y):
	peakidx, prop = find_peaks(-y, prominence=0.05)
	peakx = x[peakidx]
	peaky = y[peakidx]

	return peakx, peaky


def get_maxima(x, y):
	peakidx, prop = find_peaks(y, prominence=0.05)
	peakx = x[peakidx]
	peaky = y[peakidx]

	return peakx, peaky


def cut_spectra(x, y, a, b):
	ai = get_wavelenght_index(x, a)
	bi = get_wavelenght_index(x, b)
	return x[ai:bi+1], y[ai:bi+1]


def plot_time_resolved(t, x, y):
	plt.imshow(y, 
				aspect='auto', origin='lower', 
				extent=[t.min(), t.max(), x.min(), x.max()],
				cmap='hot')
	plt.xlabel('t (s)')
	plt.ylabel('$\lambda$ (nm)')


def plot_kinetics(ts, avgs, fits=None, labels=None):
	plt.xlabel('t (s)')
	plt.ylabel('Absorption (a.u.)')
	N = len(ts)
	for i in range(N):
		if labels is None:
			label = None
		else:
			label = labels[i]

		plt.scatter(ts[i], avgs[i], label=label)
		
		if fits is not None:
			plt.plot(ts[i], fits[i])

	if not labels is None: plt.legend()


def fit_kinetics(t, y, model='linear'):
	if model == 'exp':
		f = lambda x, *args: args[0] * (1-np.exp(-x/args[1])) + args[2]
		p0 = [1, 200, 0 ]
	if model == 'linear':
		f = lambda x, *args: args[0] * x + args[1]
		p0 = [0, 0]

	popt, pconv = curve_fit(f, t, y, p0)

	if model == 'exp':
		print(f'Exponential fit done')
		print(f'Lifetime  = {popt[1]:.2f} s')
	if model == 'linear':
		print(f'Linear fit done')
		print(f'Slope     = {popt[0]:.6f} a.u./s')
		print(f'Intercept = {popt[1]:.6f} a.u.')

	return f(t, *popt), popt



files = ['20211210_50um_no_au_DASA_10s_no_pump_06OD_filter/data.csv', 
		 '20211210_50um_no_au_DASA_20s_no_pump_06OD_filter/data.csv',
		 '20211210_50um_no_au_DASA_30s_no_pump_06OD_filter/data.csv',]
dts = [10, 20, 30]

avgs = []
ts = []
slopes = []
fits = []

for file, dt in zip(files, dts):

	data = file_reader(file)

	x = data[:,0]
	y = data[:,1:]
	N = y.shape[1]

	t = np.arange(N) * dt
	ts.append(t)

	avg545 = get_average_signal_between(x, y, 545-20, 545+20)
	avgs.append(avg545)

	fit, popt = fit_kinetics(t, avg545, model='linear')
	slopes.append(popt[0])
	fits.append(fit)

plt.figure()
plt.subplot(1,2,1)
plot_kinetics(ts, avgs, fits=fits, labels=[rf'$\Delta t$ = {dt}s' for dt in dts])

plt.subplot(1,2,2)
plt.scatter(dts, [1e5*s for s in slopes])
plt.ylabel('Slope $10^5$(a.u./s)')
plt.xlabel(r'$\Delta t$ (s)')

plt.show()