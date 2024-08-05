import json
from sys import argv

assert len(argv) == 2, 'Incorrect input format'
person = int(argv[1])

datafile = f"data/crowley/crowley{person:02d}.pickle"

with open('QML_crowley_input.dat', 'r') as f:
    d = json.load(f)
    d['datafile'] = datafile

with open('QML_crowley_input.dat', 'w') as f:
    json.dump(d, f, indent=0)
