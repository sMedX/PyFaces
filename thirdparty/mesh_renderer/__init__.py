__author__ = 'Ruslan N. Kosarev'

from mesh_renderer.mesh_renderer import mesh_renderer


def initialize(camera, light, points, cells, normals, colors, width, height):

    # initialize renderer
    renderer = mesh_renderer(
        points,
        cells,
        normals,
        colors,
        camera.position,
        camera.look_at,
        camera.up,
        light.positions,
        light.intensities,
        width,
        height,
        ambient_color=camera.ambient_color.tensor,
        fov_y=camera.fov_y,
        near_clip=camera.near_clip,
        far_clip=camera.far_clip
    )

    return renderer
