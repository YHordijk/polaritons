import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



# x = np.linspace(0, 3600, 360)


# slopes = []
# t_error = []
# for slope in np.linspace(0,.0004, 10):
# 	p = [5, 300, slope]
# 	y = p[0] * np.exp(-x/p[1]) - p[2]*x

# 	popt, pconv = curve_fit(lambda x, *p:p[0]*np.exp(-x/p[1]), x, y, [p[0], p[1]])
# 	# plt.scatter(x, y)
# 	# plt.plot(x, popt[0]*np.exp(-x/popt[1]))
# 	# plt.plot(x, y-popt[0]*np.exp(-x/popt[1]))
# 	t_error.append(p[1]-popt[1])
# 	slopes.append(slope)

# plt.plot(slopes, t_error)
# plt.show()




ftrue = lambda x, *p: p[0]*np.exp(-x/p[2]) + p[1]*np.exp(-x/p[3])
fmodel = lambda x, *p: p[0]*np.exp(-x/p[1]) + p[2]*x + p[3]


x = np.linspace(0, 2000, 360)
y = ftrue(x, *[1, 0.2, 300, 2000])
popt, pconv = curve_fit(fmodel, x, y, [1, 300, 0, 0])
plt.scatter(x, y)
plt.plot(x, fmodel(x, *popt), color='red')
plt.show()

plt.plot(x, y-fmodel(x, *popt))
print(popt)
plt.show()