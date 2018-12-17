__author__ = 'Ruslan N. Kosarev'

import os
import json

inpdir = os.path.join(os.path.pardir, 'data')

# ======================================================================================================================
if __name__ == '__main__':

    # read json file
    filename = os.path.join(inpdir, 'subject_01/Model/frontal1/obj/110920150452.json')
    with open(filename) as f:
        data = json.load(f)

    print(data.keys())

    # points
    print(data['vertices'][0]['name'])
    points = data['vertices'][0]['values']

    # normals
    print(data['vertices'][1]['name'])
    normals = data['vertices'][1]['values']

    # colors
    print(data['vertices'][2]['name'])
    colors = data['vertices'][2]['values']

    # cells
    print(data['connectivity'][0]['name'])
    cells = data['connectivity'][0]['indices']

