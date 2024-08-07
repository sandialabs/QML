import pickle 
from sklearn import datasets
from isomap import plot_3d, get_isomap

n_samples = 2000
#S_points, S_color = datasets.make_s_curve(n_samples, random_state=10)
S_points, S_color = datasets.make_swiss_roll(n_samples, random_state=10)

with open('data/swiss2000.pickle', 'rb') as f:
    points = pickle.load(f)

with open('data/swiss2000Colored.pickle', 'rb') as f:
    colors = pickle.load(f)



plot_3d(points, colors, '3D Swiss Roll')
get_isomap(points, colors, title='Swiss Roll 2D Isomap, neighbors=12')