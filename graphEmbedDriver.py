import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
    datafile = 'bioGraph.pickle'
    data = np.load(datafile, allow_pickle=True)
    colorfile = 'bioGraphColored.pickle'
    color = np.load(colorfile, allow_pickle=True)
    colorfile2 = 'bioGraphColored2.pickle'
    color2 = np.load(colorfile2, allow_pickle=True)
    colorToLabel = {}
    with open('bioGraphLabels.pickle', 'rb') as f:
        colorToLabel = pickle.load(f)
        print("color to label", colorToLabel)
    graphfile = 'bioGraph.txt'

    # # for after QML
    # datafile = '../QML_test_input.dat.out'
    # data = np.loadtxt(datafile, delimiter=",", dtype=np.double)
    # data[data == 0.0] = np.inf
    # data = data * 100;
    # graphfile = 'qmlOut.txt'

    # # connected component analysis and reduction to largest
    # # Find connected components
    # n_components, component_labels = connected_components(data, directed=False)
    # # Find the largest connected component
    # largest_component_label = np.argmax(np.bincount(component_labels))
    # # Extract the largest connected component indices
    # largest_component_indices = np.where(component_labels == largest_component_label)[0]
    # # Reduce the matrix to only nodes in the largest connected component
    # largest_connected_component = data[largest_component_indices][:, largest_component_indices]
    # largest_connected_values = color[largest_component_indices]
    # print("Largest Connected Component:")
    # print(largest_connected_component)
    # data = largest_connected_component
    # color = largest_connected_values
    # print("color", color, color.shape)

    label = []
    label2 = []
    for i in range(color.shape[0]):
        # print("color i ", color[i])
        label.append(colorToLabel[int(color[i])])
        label2.append(colorToLabel[int(color2[i])])
    print("label ", label)
    print("label2 ", label2)

    labelfile = 'bioGraphLabels.csv'
    lF = pd.DataFrame(label)
    lF.to_csv(labelfile, header=False, index=False)

    labelfile2 = 'bioGraphLabels2.csv'
    lF2 = pd.DataFrame(label2)
    lF2.to_csv(labelfile2, header=False, index=False)

    print(data)
    print(data.shape)
    # graphfile = 'bioGraph.txt'
    with open(graphfile, "w") as file:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (not np.isinf(data[i,j])):
                # if (not data[i,j] == 0.0):
                    print(i, j, data[i,j], file=file)
                    # print(i, j, data[i,j]*100, file=file)
                    # print(i, j, (1-(data[i,j]) + 0.1)*10, file=file)
                    # print(i, j, 1.0, file=file)

    colorfile = 'bioGraphColored.csv'
    DF = pd.DataFrame(color)
    # save the dataframe as a csv file
    DF.to_csv(colorfile, header=False, index=False)
    # with open(colorfile, "w") as file:
    #     for i in range(color.shape[0]):
    #         # print(str(color[i]).replace(' [', '').replace('[', '').replace(']', '').replace('.',''))
    #         print(str(color[i]).replace(' [', '').replace('[', '').replace(']', '').replace('.',''), ", ", file=file)


    colorfile2 = 'bioGraphColored2.csv'
    DF = pd.DataFrame(color2)
    # save the dataframe as a csv file
    DF.to_csv(colorfile2, header=False, index=False)


    # data[np.isinf(data)] = 0
    # G = nx.Graph(np.array(data))
    # nx.draw(G, pos = nx.spectral_layout(G))
    # plt.show()