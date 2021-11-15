import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
try:
	from pkg.lorentz_fitter import fit_lorentzian
except ModuleNotFoundError:
	from lorentz_fitter import fit_lorentzian



def get_FSR(peakx, lowx=4000):
	peakx = peakx[peakx > lowx]
	diff = -np.diff(peakx)
	diff = diff[np.abs(diff-diff.mean()) < 20] #removelarge errors
	FSR = np.mean(diff) 
	offset = peakx[-1]%FSR
	return {'FSR':FSR, 'offset':offset}


def get_peaks(spectrax, spectray, prominence=0, debug=False):
	#get rough peak_positions
	peak, _ = scipy.signal.find_peaks(spectray, prominence=prominence, height=0)
	peaky_rough = spectray[peak]
	peakx_rough = spectrax[peak]

	FSR_rough = get_FSR(peakx_rough, 4000)['FSR']

	peaks = []
	FWHM = []
	for px, py in zip(peakx_rough, peaky_rough):
		#get peaks between peakx - FSR/2 and peakx + FSR/2
		peakxfitidxs = get_closest_index(spectrax, px+FSR_rough/4), min(get_closest_index(spectrax, px-FSR_rough/4), spectrax.shape[0])
		fitx = spectrax[peakxfitidxs[0]:peakxfitidxs[1]]
		fity = spectray[peakxfitidxs[0]:peakxfitidxs[1]]
		#get better fit
		fit_res = fit_lorentzian(fitx, fity)

		w, A, P0, B = fit_res['w'], fit_res['A'], fit_res['P0'], fit_res['B']
		
		peaks.append((fit_res['P0'], fit_res['ymax']))
		FWHM.append(fit_res['w'])

		if debug:
			plt.scatter(fitx, fity, label='exp')
			L = lambda x, w, A, P0, B: A*(1+((x-P0)/(w/2))**2)**-1 + B
			plt.plot(fitx, L(fitx, w, A, P0, B), label='pred')
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
			return
		if y <= target <= next_y:
			return y, next_y


def get_closest_index(data, target):
	#gets index of element in data that is closest to target
	return np.argsort(abs(data-target))[0]

