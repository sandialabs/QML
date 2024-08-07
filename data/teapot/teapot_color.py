import sys
from math import sqrt
import numpy as np
import pickle

import matplotlib.cm as cm
from matplotlib.colors import Normalize


cmap = cm.magma
norm = Normalize(vmin=0, vmax=156)
def dists_to_colors_mpl(dists):
    return [cmap(norm(d)) for d in dists]

# box normalization to match color spectrum
def dists_to_colors(dists, num_colors=10, make_int=False):
    least = min(dists)

    colors = []
    for d in dists:
        colors.append(d - least)

    greatest = max(colors)
    for i in range(len(colors)):
        colors[i] = colors[i] * (num_colors - 1) / greatest
    

    if make_int:
        return [int(c) + 1 for c in colors]

    return [c + 1 for c in colors]


if __name__ == '__main__':
    MAX = 90;
    ANG_PER_DIM = 10;
    inc = MAX/ANG_PER_DIM

    distances = []
    for x in range(ANG_PER_DIM):
        for y in range(ANG_PER_DIM):
            for z in range(ANG_PER_DIM):
                distances.append(sqrt(x*x + y*y + z*z))

    print(min(distances), max(distances))
    print(np.std(distances))

    if len(sys.argv) > 1:
        color_info = sys.argv[1]
        if color_info == 'mpl':
            num_colors = 'mpl'
            colors = dists_to_colors_mpl(distances)
        else:
            num_colors = int(sys.argv[1])
            colors = dists_to_colors(distances, num_colors)
    else:
        num_colors = 10
        colors = dists_to_colors(distances, num_colors)
    

    print(colors)
    with open(f'teapot_colors_{num_colors}.pickle', 'wb') as f:
        pickle.dump(np.array(colors), f)
