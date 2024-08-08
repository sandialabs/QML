from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
from hover import enable_hover

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    sc = ax.scatter(x, y, c=points_color, s=50, alpha=0.8, cmap='magma')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    return sc

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8, cmap='magma')
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title, hover_data=None):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    sc = add_2d_scatter(ax, points, points_color)

    if hover_data:
       enable_hover(hover_data, fig, ax, sc) 
    plt.show()

def get_isomap(points, colors, n_neighbors=12, n_components=2, title='Isomap Embedding', hover_data=False):
    isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
    start_time = time.time()
    emb = isomap.fit_transform(points)
    print(f'Time taken: {time.time() - start_time}')
    plot_2d(emb, colors, title, hover_data=hover_data)
    return emb




   

