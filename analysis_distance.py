import numpy 
from pkg.peak_finding import *
import matplotlib.pyplot as plt

n = 1.42662 #cyclohexane CHXE
n = 1.3441 #ACN
n = 1.4507 #cyclohexanone CHXO
l = lambda FSR, n: 10**4/(2*n*FSR) #in um



if __name__ == '__main__':
	file = "data/20211122_cyclohexane_distance_coupling/data.csv"

	file = "data/20211124_spacing_measurements_10SP_90ACN/data.csv"
	plotfile = "results/spacing_measurements/10SP_90ACN.png"

	# file = "data/20211124_spacing_measurements_100SP/data.csv"
	# plotfile = "results/spacing_measurements/100SP.png"

	file = "data/20211126_spacing_measurements_100SP_5um/data.csv"
	plotfile = "results/spacing_measurements/100SP_5um.png"

	low_wn = 1500
	high_wn = 1900


	spectra, names, types = read_ir(file)
	spectrax = spectra[:,0]
	spectray = spectra[:,1:].T

	spectray = 10**(-spectray) #convert to transmittance
	FSRs = []
	FSRoffsets = []

	print(f'Loaded {spectray.shape[0]} spectra!')


	for i, spec in enumerate(spectray):
		peaks_results = get_peaks(spectrax, spec, prominence=0.05)
		FSRres = get_FSR(peaks_results['peakx'], 4000)
		FSR = FSRres['FSR']
		FSRoffsets.append(FSRres['offset'])
		print(f'Spectrum {i}: FSR = {FSR:.1f} cm^-1, L = {l(FSR, n):.2f} um')
		FSRs.append(FSR)


	ls = np.array([l(FSR, n) for FSR in FSRs]) #um
	
	ls_sortidx = np.argsort(ls)
	spectray = spectray[ls_sortidx]
	ls = ls[ls_sortidx]

	FSRs = np.asarray(FSRs)[ls_sortidx]
	FSRoffsets = np.asarray(FSRoffsets)[ls_sortidx]
	FSR_lines = [[i*FSR + o for FSR, o in zip(FSRs, FSRoffsets)] for i in range(14)]

	cmap = plt.get_cmap('jet')
	for i, s in enumerate(spectray):
		plt.plot(spectrax, s, color=cmap(i/len(FSRs)))

	plt.figure()
	#cut out desired wns
	lowidx = np.argmin(abs(spectrax - low_wn))
	highidx = np.argmin(abs(spectrax - high_wn))

	#normalize them 
	cut_spec = spectray[:, highidx:lowidx]
	norm_spec = cut_spec/cut_spec.max(axis=0)
	norm_spec = cut_spec

	x, y = np.meshgrid(ls, np.linspace(high_wn, low_wn, norm_spec.shape[1]))

	# plt.title('Spectral evolution with changing cavity size')
	plt.ylabel('Wavenumber ($cm^{-1}$)')
	plt.xlabel('Spacing ($\mu m$)')
	plt.ylim((low_wn, high_wn))
	plt.xlim((ls.min(), ls.max()))

	plt.pcolormesh(x, y, norm_spec.T, cmap='jet', shading='gouraud')
	plt.plot((ls.min(), ls.max()), (1720,1720), linestyle='dashed', color='black')

	# for FSR_line in FSR_lines:
	# 	plt.plot(ls, FSR_line, color='red', linewidth=4)

	plt.tight_layout()
	plt.savefig(plotfile)
	plt.show()