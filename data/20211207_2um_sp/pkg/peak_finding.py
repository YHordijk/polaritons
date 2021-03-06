import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
try:
	from pkg.lorentz_fitter import fit_lorentzian
except ModuleNotFoundError:
	from lorentz_fitter import fit_lorentzian



def read_ir(file):
	with open(file, 'r') as f:
		lines = f.readlines()
		spectrum_names = lines[0].strip().split(',')[1:][:-1]
		spectrum_types = lines[1].lower().strip().strip(',').split(',')[1:][:-1]
	data = np.genfromtxt(file, skip_header=2, delimiter=',')[:,:-1]  #dont send last row cause itis empty
	return data, spectrum_names, spectrum_types


def read_UV(file):
	data = np.genfromtxt(file, skip_header=1, delimiter=',')[:,:-1]
	return data



def get_FSR(peakx, lowx=4000):
	peakx = peakx[peakx > lowx]
	diff = np.abs(np.diff(peakx))
	diff = diff[np.abs(diff-diff.mean()) < 20] #removelarge errors
	FSR = np.mean(diff)
	if np.isnan(FSR):
		return
	offset = peakx[-1]%FSR
	if abs(offset-FSR) < offset:
		offset = offset-FSR
	return {'FSR':FSR, 'offset':offset}


def get_peaks(spectrax, spectray, prominence=0, debug=False):
	peak, props = scipy.signal.find_peaks(spectray, prominence=prominence, height=0, width=0)
	peaky_rough = spectray[peak]
	peakx_rough = spectrax[peak]
	delta = abs(np.mean(np.diff(spectrax)))
	left_points = props['left_ips']*delta  + spectrax.min()
	right_points = props['right_ips']*delta  + spectrax.min()

	peaks = []
	FWHM = []
	for px, py, l, r in zip(peakx_rough, peaky_rough, left_points, right_points):
		#get peaks between peakx - FSR/2 and peakx + FSR/2
		peakxfitidxs = [get_closest_index(spectrax, px-30), get_closest_index(spectrax, px+30)]
		peakxfitidxs.sort()
		fitx = spectrax[peakxfitidxs[0]:peakxfitidxs[1]]
		fity = spectray[peakxfitidxs[0]:peakxfitidxs[1]]
		#get better fit
		try:
			fit_res = fit_lorentzian(fitx, fity)
			w, A, P0, B = fit_res['w'], fit_res['A'], fit_res['P0'], fit_res['B']
			peaks.append((fit_res['P0'], fit_res['ymax']))
			FWHM.append(fit_res['w'])
		except:
			peaks.append((px,py))
			FWHM.append(1)
			# raise

		if debug:
			plt.plot(spectrax, spectray)
			plt.scatter(fitx, fity, label='exp')
			L = lambda x, w, A, P0, B: A*(1+((x-P0)/(w/2))**2)**-1 + B
			plt.plot(fitx, L(fitx, w, A, P0, B), label='pred')
			plt.scatter(P0, L(P0, w, A, P0, B))
			plt.legend()
			plt.show()


	results = {'peakx': np.asarray([p[0] for p in peaks]), 'peaky': np.asarray([p[1] for p in peaks]), 'FWHM': np.asarray(FWHM)}
	#get peak_height
	return results


def get_adjacent_peaks(target, ys):
	#gets values in ys that are left and right of value target
	ys = np.sort(ys)
	
	for i, y in enumerate(ys):
		try:
			next_y = ys[i+1]
		except IndexError:
			raise
		if y <= target <= next_y:
			return y, next_y


def get_closest_index(data, target):
	#gets index of element in data that is closest to target
	return np.argsort(abs(data-target))[0]


def get_average_signal(x, y, low, high):
	#gets the average signal height in spectrum x, y
	#between low and high on x
	#|-l-|---|---|---|---|---|---|h--|
	#i                               j

	#start out with getting i and j
	i = get_closest_index(x, low)
	j = get_closest_index(x, high)
	#ensure that x[i] < low and x[j] > high
	if not x[i] <= low:
		i -= 1
	if not x[j] >= high:
		j += 1

	#get sum over these values
	s = np.sum(y[i:j+1])

	#correct sum for low and high ends
	Xl = (low - x[i])/(x[i+1] - x[i])
	Xh = (high - x[j-1])/(x[j] - x[j-1])

	Yl = Xl * y[i+1] + (1 - Xl) * y[i]
	Yh = Xh * y[j] + (1 - Xh) * y[j-1]

	Al = (y[i] + abs(Yl-y[i])/2) * (low - x[i])
	Ah = (y[j] + abs(y[j]-high)/2) * (x[j] - high)
	print(s)
	s = s - Al - Ah
	# print(x[i], low, high, x[j])
	if high == low:
		return 0
	return s/(high-low)

if __name__ == '__main__':
	x = np.linspace(0, 100, 1100)
	y = np.sin(x)
	highs = np.linspace(0,10,100)
	avgs = []
	for high in highs:
		avgs.append(get_average_signal(x, y, 0, high))

	plt.plot(highs, avgs)
	plt.show()