import pickle 
from sklearn import datasets
from isomap import get_isomap
from math import sqrt


with open('data/teapot/teapot_resized02.pickle', 'rb') as f:
    points = pickle.load(f)

with open('data/teapot/teapot_colors_mpl.pickle', 'rb') as f:
    colors = pickle.load(f)


MAX = 90;
ANG_PER_DIM = 10;
inc = MAX/ANG_PER_DIM

distances = []
for x in range(ANG_PER_DIM):
    for y in range(ANG_PER_DIM):
        for z in range(ANG_PER_DIM):
            distances.append(sqrt(x*x + y*y + z*z))

get_isomap(points, distances, title='Teapot 2D Isomap, neighbors=12')