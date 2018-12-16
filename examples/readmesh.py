__author__ = 'Ruslan N. Kosarev'

import os
import thirdparty.pywavefront as pw

inpdir = os.path.join(os.path.pardir, 'data')


# ======================================================================================================================
if __name__ == '__main__':
    # pyglet.clock.schedule(update)

    filename = os.path.join(inpdir, 'subject_01/Model/frontal1/obj/110920150452.obj')

    obj = pw.Wavefront(filename)

    vertices = obj.vertices

    # Iterate vertex data collected in each material
    for name, material in obj.materials.items():
        # Contains the vertex format (string) such as "T2F_N3F_V3F"
        # T2F, C3F, N3F and V3F may appear in this string
        print(material.vertex_format)
        # Contains the vertex list of floats in the format described above
        print(material.vertices)

        # Material properties
        print(material.diffuse)
        print(material.ambient)
        print(material.texture)

