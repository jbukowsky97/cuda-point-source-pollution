import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

temp = []

with open('single-heatmap.txt') as file:
	# for timestep in file.split('-'):
		# pprint(timestep)
	print(type(file))
    # array = [[float("{:.3f}".format(float(digit))) for digit in line.split(',')] for line in file]

np_array = np.array(array)
plt.imshow(np_array, cmap='hot', interpolation='nearest')
plt.show()




