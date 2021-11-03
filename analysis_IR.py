import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def read_IR(file):
	data = np.genfromtxt(file, skip_header=2, delimiter=',')[:,:-1]  #dont send last row cause itis empty
	return data

def get_peaks(absorbances):
	peaks = []
	for absor in absorbances:
		print(absor)
		peak, _ = scipy.signal.find_peaks(absor)
		peaks.append(peak)
	return peaks

if __name__ == '__main__':
	file = 'data/20211102 IRs acetone cavity/all_Irs.csv'

	data = read_IR(file)
	wavelenghts = data[:,0]
	absorbances = data[:,1:]

	cmap = plt.get_cmap('jet')
	N_spectra = absorbances.shape[1]

	print(f'Found {N_spectra} spectra! λ ⊂ [{wavelenghts[0]} nm, {wavelenghts[-1]} nm]')

	plt.figure()
	plt.title('Spectra')
	plt.xlabel('λ (nm)')
	plt.ylabel('Absorbance')
	for i, absorbance in enumerate(absorbances.T):
		plt.plot(wavelenghts, absorbance , c=cmap(i/N_spectra))

	#calculations
	#Rabi splitting
	peaks = get_peaks(absorbances)
	print(peaks)

	plt.show()
