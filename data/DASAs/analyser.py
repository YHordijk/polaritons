import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks 


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



file = "20211209_cuvette_DASA_3inone/data.csv"

data = file_reader(file)

spectrax = data[:,0]
spectray = data[:,1:]
N = spectray.shape[1]
cmap = plt.get_cmap('viridis')

for i, y in enumerate(spectray.T):
	plt.plot(spectrax, y, color=cmap(i/N))


plt.figure()
plt.subplot(1,2,1)
plt.suptitle('Time-resolved spectra and kinetics\nDASA 1 in 1cm cuvette')

t = np.arange(N) * 10
l545 = get_average_signal_between(spectrax, spectray, 545-10, 545+10)
l600 = get_average_signal_between(spectrax, spectray, 590, 610)

kinetic_starts = [140, 430, 780, 1180]
kinetic_ends = [350, 720, 1070, 1600]

plt.xlabel('t (s)')
plt.ylabel('Absorption (a.u.)')

plt.plot(t, l545, label='Signal evolution $\lambda = 545 nm$')
plt.plot(t, l600, label='Signal evolution $\lambda = 600 nm$')
plt.vlines(kinetic_starts, [0 for _ in kinetic_starts], [l545.max() for _ in kinetic_starts], colors='black', linestyles='dashed')
plt.vlines(kinetic_ends, [0 for _ in kinetic_ends], [l545.max() for _ in kinetic_ends], colors='black', linestyles='dashed')
for i, a, b in zip(range(len(kinetic_starts)), kinetic_starts, kinetic_ends):
	if i == 0:
		plt.fill_between([a, b], 0, l545.max(), alpha=0.1, color='purple', label='Decay area')
	else:
		plt.fill_between([a, b], 0, l545.max(), alpha=0.1, color='purple')
plt.legend()

plt.subplot(1,2,2)
imshow_l_range = [290, 750]
idx = [get_wavelenght_index(spectrax, l) for l in imshow_l_range]
plt.imshow(spectray[idx[0]:idx[1]], 
			aspect='auto', origin='lower', 
			extent=[t.min(), t.max(), imshow_l_range[0], imshow_l_range[1]],
			cmap='hot')
plt.xlabel('t (s)')
plt.ylabel('$\lambda$ (nm)')

for point, label in zip([600, 545, 567], ["A'", 'A', 'Isosbestic']):
	plt.hlines(point, 0, t.max(),  
				linestyles='dashed', 
				label=label,)

plt.legend()
plt.show()