import imageio

def build_gif():
	images = []
	for filename in filenames:
	    images.append(imageio.imread(filename))
	imageio.mimsave('/path/to/movie.gif', images)

build_gif()