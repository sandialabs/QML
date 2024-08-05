import sys
import glob
import os
import json
import pickle
from data.crowley.conv_images import image_from_vector
from qml_serial_preprocess import initialize

# Main visualization logic: displays each of nProp propagations starting from the START_IDX-th image
# SHAPE is the shape of each image: image_size = SHAPE[0] * SHAPE[1]
# x is the original, un-normalized condensed dataset matrix: N x image_size
# peak_idxs is the same as in qml_serial --- the propagation images for each start image: [nProp x nColl] x N
def visualize_propagations(x, peak_idxs, nProp, nColl, START_IDX, SHAPE):
    curr_props = peak_idxs[START_IDX]
    print(curr_props)

    #last = nProp//4
    last = nProp
    for coll_idx in range(nColl):
        for prop_row in range(last):
            point = curr_props[prop_row][coll_idx]
            fname = f'prop_s{coll_idx}_t{prop_row}_c{point}.jpg'
            assert x[point].shape[0] == SHAPE[0] * SHAPE[1], 'Incorrect shape for image vector'
            image_from_vector(x[point], SHAPE)

# retrieves the latest directory in the results directory
def last_result():
    all_result_dirs = glob.glob(os.path.join(RESULTS_DIR, '*'))
    return max(all_result_dirs, key=os.path.getctime)

RESULTS_DIR = 'results'

if __name__ == '__main__':
    dir_name = sys.argv[1] if len(sys.argv) > 1 else last_result()
    start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 3


    if not os.path.isdir(dir_name):
        dir_name = os.path.join(RESULTS_DIR, dir_name)

    for fname in os.listdir(dir_name):
        filename = os.fsdecode(fname)
        if filename.endswith('.dat'):
            dat_file = os.path.join(dir_name, filename)
            break
    try:
        f = open(dat_file)
    except:
        print("Cannot open input file: " + sys.argv[1] + "... Exiting.")
    else:
        data = f.read()
        inp = json.loads(data)

        # initialize qml_params
        qml_params = initialize(inp)

        print(qml_params)
        print('\n')

        nProp = qml_params['nProp']
        nColl = qml_params['nColl']

        # gets the shape of the image
        with open(os.path.join(dir_name, 'shape.txt'), 'r') as f:
            shape_lines = f.readlines()
            SHAPE = (int(shape_lines[0]), int(shape_lines[1]))        
        
        # gets the condensed dataset
        with open(os.path.join(dir_name, 'x.pickle'), 'rb') as f:
            x = pickle.load(f)

        # gets the propagation information
        with open(os.path.join(dir_name, 'peak_idxs.pickle'), 'rb') as f:
            peak_idxs = pickle.load(f)

        visualize_propagations(x, peak_idxs, nProp, nColl, start_idx, SHAPE)

        


