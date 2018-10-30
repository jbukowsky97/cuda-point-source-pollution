#!/usr/bin/env python
from __future__ import print_function
import os,sys
import imageio

def build_gif(filenames):
	images = []
	for filename in filenames:
	    images.append(imageio.imread('gif-images/' + filename))
	imageio.mimsave('/home/dante/Documents/school/cuda-point-source-pollution/2D/gif-images/heatmap.gif', images)

#set path for files
path = '/home/dante/Documents/school/cuda-point-source-pollution/2D/gif-images'
if len(sys.argv) == 2:
    path = sys.argv[1] 
 
#get all files in the dir
files = os.listdir(path)

#get files into dict
file_dict = {}
for name in files:
    number = int(name.split('h')[0])
    file_dict[number] = name

#get files into ordered list
ordered_files = []
for i in range(len(file_dict)):
	ordered_files.append(file_dict[i])

#make the gif
build_gif(ordered_files)