from PIL import Image
import numpy as np
import os
import pickle

SHAPE = (720, 819)

# options for type: 'L', 'LA', 'RGB'
def matrix_from_image(fname, type='L'):
	image = Image.open(fname).convert(type)
	return np.asarray(image)

def image_from_vector(vector, shape, fname=None, type='L'):
	m = vector.reshape(shape)
	image = Image.fromarray(np.asarray(m).astype(np.uint8))
	if fname:
		image.save(fname)
	
	image.show()
	return image


def resize_images_in_dir(dir_name, new_size):
	directory = os.fsencode(dir_name)
	resized_dir = f'resized_{new_size}'.replace(' ', '')
	os.makedirs(resized_dir)
	for fname in os.listdir(directory):
		filename = os.fsdecode(fname)
		if filename.endswith(".jpg") or filename.endswith(".png"):
			full_fname = os.path.join(directory.decode('utf-8'), filename)
			im = Image.open(full_fname).resize(new_size)
			new_full_fname = os.path.join(resized_dir, filename)
			im.save(new_full_fname)
			
	


# outputs a list of matrices where each matrix corresponds to an image in the directory
def matrices_in_dir(dir_name):
	matrices = []
	directory = os.fsencode(dir_name)
	for fname in os.listdir(directory):
		filename = os.fsdecode(fname)
		if filename.endswith(".jpg") or filename.endswith(".png"):
			full_fname = os.path.join(directory.decode('utf-8'), filename)
			matrices.append(matrix_from_image(full_fname))
	return matrices


# converts from a dataset (set of image vectors) pickle file to a matrix
# if image_num isn't none, then it converts to that specific image vector and reshapes
def decode_pickle(pickle_file, shape=SHAPE, image_num=None):
	with open(pickle_file, 'rb') as f:
		m = pickle.load(f)
		if image_num:
			return m[image_num].reshape(shape)
		return m



def main():
	dir_names = ['resized_(205,180)', 'resized_(102,90)']
	# num_starting_teapots = 1
	# for i in range(1,num_starting_teapots+1):
	# 	s = f'teapot{i:02d}'
	# 	dir_names.append(s)


	# 1 collections for 1 teapot
	# Each collection has 1000 matrices (1000 images)
	# Each matrix is 720 x 819 for L (grayscale)
	# Each matrix is 720 x 819 x 2 for LA (grayscale + alpha)
	# Each matrix is 720 x 819 x 3 for RGB (Red Green Blue)
	all_matrices = [matrices_in_dir(d) for d in dir_names]
	print(len(all_matrices))
	print(len(all_matrices[0]))

	# converts collection of images into one matrix per person (each image becomes a vector)
	all_matrices_condensed = [np.array([mat.reshape(mat.size) for mat in dataset]) 
								for dataset in all_matrices]
							
	for dataset_index, matrix in enumerate(all_matrices_condensed):
		filename = f'teapot_resized{ dataset_index + 1 :02d}.pickle'
		with open(filename, 'wb') as file:
			pass
			#pickle.dump(matrix,file)

	return all_matrices_condensed


if __name__ == '__main__':
	main()