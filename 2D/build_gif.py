import imageio

def build_gif(filenames):
	images = []
	for filename in filenames:
	    images.append(imageio.imread('gif-images/' + filename))
	imageio.mimsave('/home/dante/Documents/school/cuda-point-source-pollution/2D/gif-images/heatmap.gif', images)


filenames = ['heatmap0.png','heatmap2.png','heatmap4.png','heatmap6.png','heatmap8.png','heatmap1.png','heatmap3.png','heatmap5.png','heatmap7.png','heatmap9.png']
build_gif(filenames)



