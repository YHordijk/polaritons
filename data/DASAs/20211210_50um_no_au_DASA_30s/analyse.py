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


def plot_kinetics(t, avgs, labels=None):
	plt.xlabel('t (s)')
	plt.ylabel('Absorption (a.u.)')
	for i, avg in enumerate(avgs):
		fit, popt = fit_kinetics(t, avg)

		if labels is None:
			label = None
		else:
			label = labels[i]

		plt.plot(t, avg, label=label + rf', $\tau$ = {popt[1]:.2f} s')
		plt.plot(t, fit)

	if not labels is None: plt.legend()


def fit_kinetics(t, y, model='exp'):
	if model == 'exp':
		f = lambda x, *args: args[0] * (1-np.exp(-x/args[1])) + args[2]

	popt, pconv = curve_fit(f, t, y, [1, 200, 0 ])
	print(f'Lifetime = {popt[1]:.2f} s')

	return f(t, *popt), popt



file = 'data.csv'

data = file_reader(file)

x = data[:,0]
y = data[:,1:]
N = y.shape[1]
t = np.arange(N) * 30

plt.figure()
plt.suptitle('Kinetics measurement every 30s with pump')
plt.subplot(1,2,2)
plot_time_resolved(t, *cut_spectra(x, y, 400, 600))

plt.subplot(1,2,1)
avg545 = get_average_signal_between(x, y, 545-10, 545+10)
# avg600 = get_average_signal_between(x, y, 600-5, 600+5)
plot_kinetics(t, [avg545], [r'$\lambda = 545 nm$'])

plt.show()