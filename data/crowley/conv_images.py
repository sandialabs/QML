from PIL import Image
import numpy as np
import os
import pickle

SHAPE = (288, 384)

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



def matrices_in_dir(dir_name):
	matrices = []
	directory = os.fsencode(dir_name)
	for fname in os.listdir(directory):
		filename = os.fsdecode(fname)
		if filename.endswith(".jpg") or filename.endswith(".png"):
			full_fname = os.path.join(directory.decode('utf-8'), filename)
			matrices.append(matrix_from_image(full_fname))
	return matrices


def decode_pickle(pickle_file, image_num=None):
	with open(pickle_file, 'rb') as f:
		m = pickle.load(f)
		if image_num:
			return m[image_num].reshape(SHAPE)
		return m



def main():
	dir_names = []
	num_persons = 12
	for i in range(1,num_persons+1):
		s = f'Person{i:02d}'
		dir_names.append(s)


	# 12 collections for 12 persons
	# Each collection has 168 matrices (168 images)
	# Each matrix is 288 x 384 for L (grayscale)
	# Each matrix is 288 x 384 x 2 for LA (grayscale + alpha)
	# Each matrix is 288 x 384 x 3 for RGB (Red Green Blue)
	all_matrices = [matrices_in_dir(d) for d in dir_names]
	print(len(all_matrices))
	print(len(all_matrices[0]))

	# converts collection of images into one matrix per person
	all_matrices_condensed = [np.array([mat.reshape(mat.size) for mat in person_mats]) 
								for person_mats in all_matrices]
							
	for person_index, matrix in enumerate(all_matrices_condensed):
		filename = f'crowley{ person_index + 1 :02d}.pickle'
		with open(filename, 'wb') as file:
			pickle.dump(matrix,file)

	return all_matrices_condensed


if __name__ == '__main__':
	main()