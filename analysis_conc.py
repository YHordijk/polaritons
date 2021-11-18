import pkg.peak_finding as pkf
import numpy as np
import matplotlib.pyplot as plt
import scipy






if __name__ == '__main__':
	file = "data/20211118_cyclohexanon_concentration_dependence/data.csv"
	polariton_wn = 1720

	#obtain data
	data = pkf.read_ir(file)
	spectrax = data[0][:, 0].T
	spectray = data[0][:, 1:].T
	names = data[1]
	concentrations = np.asarray([float(n.split('_')[0]) for n in names])
	concidx = np.argsort(concentrations)
	concentrations = concentrations[concidx]
	spectray = spectray[concidx]

	# spectray = spectray[spectrax>1000]
	# spectrax = spectrax[spectrax>1000]

	polariton_peaks = np.array([[1699.12, 1742.47],
								[1702.52, 1751.71],
								[1694.10, 1758.74],
								[1691.24, 1747.64],
								[1689.50, 1761.53],
								[1685.40, 1752.18],
								[1676.57, 1756.03],
								[1683.01, 1766.83],
								[1672.49, 1773.69],
								[1668.12, 1762.97],
								[1662.20, 1768.03],
								[1667.47, 1776.64],
								[1663.87, 1780.31],
								[1659.93, 1771.92],
								[1657.37, 1775.39],
								[1661.05, 1782.53],
								[1652.32, 1777.08],
								[1658.83, 1785.16],
								[1648.97, 1779.06],
								[1655.47, 1785.82]])

	# splits = []
	# for c, s in zip(concentrations, spectray):
		# s = s[np.logical_and(spectrax>1000, spectrax<2000)]
		# plt.plot(spectrax[np.logical_and(spectrax>1000, spectrax<2000)], s)
		# plt.show()
		# peak_res = pkf.get_peaks(spectrax[np.logical_and(spectrax>1000, spectrax<2000)], s, prominence=0)
		# # plt.scatter(peak_res['peakx'], peak_res['peaky'])
		# l, h = pkf.get_adjacent_peaks(polariton_wn, peak_res['peakx'])
		# # plt.show()
		# plt.vlines([l,h], s.min(), s.max(), colors='red', linewidths=1)

		# splitting = (h-l)*4.135667696e-15*2.99792458e10

		# splits.append(splitting)

		# print(f'Concentration {c: >3f} (V/V)% => {splitting:.3f} eV = {h-l:.2f} cm^-1')

		# plt.show()


	plt.figure()
	plt.title('Concentration dependence of cyclohexanone coupling')

	splits = []	
	splitswn = []
	for c, p in zip(concentrations, polariton_peaks):
		splitwn = float(abs(np.diff(p)))
		rabi = splitwn * 4.135667696e-15*2.99792458e10

		print(f'Concentration {c} (V/V)% => rabi = {rabi:.3f} eV, dv = {splitwn:.2f} cm^-1')
		splits.append(rabi)
		splitswn.append(splitwn)


	x = np.asarray(concentrations)**.5
	y = np.asarray(splitswn)

	plt.scatter(x, y, label='Experiment')
	plt.xlabel(r'$\sqrt{(V_{cyclohexanone}/V_{cyclohexane})\%}$')
	plt.ylabel('Rabi splitting (cm^-1)')

	linregres = scipy.stats.linregress(x, y)

	plt.plot(x, linregres.slope*x + linregres.intercept, label=f'Linear regression r = {linregres.rvalue}')
	plt.legend()
	plt.show()

