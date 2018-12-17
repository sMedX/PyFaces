__author__ = 'Ruslan N. Kosarev'

import os
import json
import numpy as np

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
    number_of_points = int(len(points) / 3)
    points = np.reshape(points, [number_of_points, 3])

    # normals
    print(data['vertices'][1]['name'])
    normals = data['vertices'][1]['values']
    normals = np.reshape(normals, [number_of_points, 3])

    # colors
    print(data['vertices'][2]['name'])
    colors = data['vertices'][2]['values']
    colors = np.reshape(colors, [int(len(colors) / 4), 4])[:, :3]

    # cells
    print(data['connectivity'][0]['name'])
    cells = data['connectivity'][0]['indices']
    number_of_cells = int(len(cells) / 3)
    cells = np.reshape(cells, [number_of_cells, 3])

