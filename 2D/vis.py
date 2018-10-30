import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pprint import pprint

#read file into list of lists
with open('single-heatmap.txt') as file:
	current_timestep = []
	all_timesteps = []
	for line in file:
		if '-' not in line:
			current_timestep.append(line[:-2])
		else:
			all_timesteps.append(current_timestep)
			current_timestep = []

#split string into floats
array = []
for timestep in all_timesteps:
	current_timestep = []
	for nums_string in timestep:
		nums = []
		for num in nums_string.split(','):
			nums.append(float(num))
		current_timestep.append(nums)
	array.append(current_timestep)

#convert to numpy array
np_array = np.array(array)

image_num = 0
for timestep in np_array:
	plt.imshow(timestep, cmap='hot', interpolation='nearest')
	plt.savefig('heatmap' + str(image_num) + '.png', bbox_inches='tight')
	image_num += 1
# plt.imshow(np_array, cmap='hot', interpolation='nearest')
# plt.show()




