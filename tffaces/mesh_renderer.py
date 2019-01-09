__author__ = 'Ruslan N. Kosarev'

import os
import inspect
import tensorflow as tf
from . import tfSession
import math

dir_name = os.path.join(os.path.dirname(inspect.stack()[0][1]), os.path.pardir, 'thirdparty/tf_mesh_renderer')
lib_path = 'bazel-out/k8-fastbuild/genfiles/mesh_renderer/kernels/rasterize_triangles_kernel.so'
module_path = os.path.abspath(os.path.join(dir_name, lib_path))
rasterize_triangles_module = tf.load_op_library(module_path)

# This epsilon should be smaller than any valid barycentric reweighting factor (i.e. the per-pixel reweighting
# factor used to correct for the effects of perspective-incorrect barycentric interpolation). It is necessary
# primarily because the reweighting factor will be 0 for factors outside the mesh, and we need to ensure
# the image color and gradient outside the region of the mesh are 0.
minimum_reweighting_threshold = 1e-6

# This epsilon is the minimum absolute value of a homogenous coordinate before it is clipped.
# It should be sufficiently large such that the output of the perspective divide step with this denominator
# still has good working precision with 32 bit arithmetic, and sufficiently small so that in practice
# vertices are almost never close enough to a clipping plane to be thresholded.
minimum_perspective_threshold = 1e-6


# ======================================================================================================================
class MeshRenderer:
    def __init__(self, vertices,
                 triangles,
                 normals,
                 diffuse_colors,
                 camera_position,
                 camera_look_at,
                 camera_up,
                 light_positions,
                 light_intensities,
                 image_width,
                 image_height,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None,
                 fov_y=40.0,
                 near_clip=0.01,
                 far_clip=10.0):
        """Renders an input scene using phong shading, and returns an output image.

        Args:
          vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
              triplet is an xyz position in world space.
          triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
              should contain vertex indices describing a triangle such that the
              triangle's normal points toward the viewer if the forward order of the
              triplet defines a clockwise winding of the vertices. Gradients with
              respect to this tensor are not available.
          normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
              triplet is the xyz vertex normal for its corresponding vertex. Each
              vector is assumed to be already normalized.
          diffuse_colors: 3-D float32 tensor with shape [batch_size,
              vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
              each vertex.
          camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
              shape [3] specifying the XYZ world space camera position.
          camera_look_at: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
              shape [3] containing an XYZ point along the center of the camera's gaze.
          camera_up: 2-D tensor with shape [batch_size, 3] or 1-D tensor with shape
              [3] containing the up direction for the camera. The camera will have no
              tilt with respect to this direction.
          light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
              XYZ position of each light in the scene. In the same coordinate space as
              pixel_positions.
          light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
              RGB intensity values for each light. Intensities may be above one.
          image_width: int specifying desired output image width in pixels.
          image_height: int specifying desired output image height in pixels.
          specular_colors: 3-D float32 tensor with shape [batch_size,
              vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
              each vertex.  If supplied, specular reflections will be computed, and
              both specular_colors and shininess_coefficients are expected.
          shininess_coefficients: a 0D-2D float32 tensor with maximum shape
             [batch_size, vertex_count]. The phong shininess coefficient of each
             vertex. A 0D tensor or float gives a constant shininess coefficient
             across all batches and images. A 1D tensor must have shape [batch_size],
             and a single shininess coefficient per image is used.
          ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
              color, which is added to each pixel in the scene. If None, it is
              assumed to be black.
          fov_y: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
              desired output image y field of view in degrees.
          near_clip: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
              near clipping plane distance.
          far_clip: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
              far clipping plane distance.

        Returns:
          A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
          containing the lit RGBA color values for each image at each pixel. RGB
          colors are the intensity values before tonemapping and can be in the range
          [0, infinity]. Clipping to the range [0,1] with tf.clip_by_value is likely
          reasonable for both viewing and training most scenes. More complex scenes
          with multiple lights should tone map color values for display only. One
          simple tonemapping approach is to rescale color values as x/(1+x); gamma
          compression is another common techinque. Alpha values are zero for
          background pixels and near one for mesh pixels.
        Raises:
          ValueError: An invalid argument to the method is detected.
        """
        if len(vertices.shape) != 3:
            raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
        self.vertices = vertices

        if len(vertices.shape) != 3:
            raise ValueError('The vertex buffer must be 3D.')
        self.triangles = triangles

        self.batch_size = vertices.shape[0].value

        if len(normals.shape) != 3:
            raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
        if len(light_positions.shape) != 3:
            raise ValueError('Light_positions must have shape [batch_size, light_count, 3].')
        if len(light_intensities.shape) != 3:
            raise ValueError('Light_intensities must have shape [batch_size, light_count, 3].')
        if len(diffuse_colors.shape) != 3:
            raise ValueError('vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')

        if not image_width > 0:
            raise ValueError('Image width must be > 0.')
        self.image_width = image_width

        if not image_height > 0:
            raise ValueError('Image height must be > 0.')
        self.image_height = image_height

        if ambient_color is not None and ambient_color.get_shape().as_list() != [self.batch_size, 3]:
            raise ValueError('Ambient_color must have shape [batch_size, 3].')
        self.ambient_color = ambient_color

        if camera_position.get_shape().as_list() == [3]:
            camera_position = tf.tile(tf.expand_dims(camera_position, axis=0), [self.batch_size, 1])
        elif camera_position.get_shape().as_list() != [self.batch_size, 3]:
            raise ValueError('Camera_position must have shape [batch_size, 3]')
        self.camera_position = camera_position

        if camera_look_at.get_shape().as_list() == [3]:
            camera_look_at = tf.tile(tf.expand_dims(camera_look_at, axis=0), [self.batch_size, 1])
        elif camera_look_at.get_shape().as_list() != [self.batch_size, 3]:
            raise ValueError('Camera_lookat must have shape [batch_size, 3]')
        self.camera_look_at = camera_look_at

        if camera_up.get_shape().as_list() == [3]:
            camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [self.batch_size, 1])
        elif camera_up.get_shape().as_list() != [self.batch_size, 3]:
            raise ValueError('Camera_up must have shape [batch_size, 3]')
        self.camera_up = camera_up

        if isinstance(fov_y, float):
            fov_y = tf.constant(self.batch_size * [fov_y], dtype=tf.float32)
        elif not fov_y.get_shape().as_list():
            fov_y = tf.tile(tf.expand_dims(fov_y, 0), [self.batch_size])
        elif fov_y.get_shape().as_list() != [self.batch_size]:
            raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with shape [batch_size]')
        self.fov_y = fov_y

        if isinstance(near_clip, float):
            near_clip = tf.constant(self.batch_size * [near_clip], dtype=tf.float32)
        elif not near_clip.get_shape().as_list():
            near_clip = tf.tile(tf.expand_dims(near_clip, 0), [self.batch_size])
        elif near_clip.get_shape().as_list() != [self.batch_size]:
            raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor with shape [batch_size]')
        self.near_clip = near_clip

        if isinstance(far_clip, float):
            far_clip = tf.constant(self.batch_size * [far_clip], dtype=tf.float32)
        elif not far_clip.get_shape().as_list():
            far_clip = tf.tile(tf.expand_dims(far_clip, 0), [self.batch_size])
        elif far_clip.get_shape().as_list() != [self.batch_size]:
            raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor with shape [batch_size]')
        self.far_clip = far_clip

        if specular_colors is not None and shininess_coefficients is None:
            raise ValueError('Specular colors were supplied without shininess coefficients.')

        if shininess_coefficients is not None and specular_colors is None:
            raise ValueError('Shininess coefficients were supplied without specular colors.')

        if specular_colors is not None:
            # Since a 0-D float32 tensor is accepted, also accept a float.
            if isinstance(shininess_coefficients, float):
                shininess_coefficients = tf.constant(shininess_coefficients, dtype=tf.float32)
            if len(specular_colors.shape) != 3:
                raise ValueError('The specular colors must have shape [batch_size, vertex_count, 3].')
            if len(shininess_coefficients.shape) > 2:
                raise ValueError('The shininess coefficients must have shape at most [batch_size, vertex_count].')
            # If we don't have per-vertex coefficients, we can just reshape the input shininess to broadcast later,
            # rather than interpolating an additional vertex attribute:
            if len(shininess_coefficients.shape) < 2:
                self.vertex_attributes = tf.concat([normals, vertices, diffuse_colors, specular_colors], axis=2)
            else:
                self.vertex_attributes = tf.concat([normals, vertices, diffuse_colors, specular_colors,
                                               tf.expand_dims(shininess_coefficients, axis=2)], axis=2)
        else:
            self.vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

        self.normalized_device_coordinates = None
        self.image_coordinates = None

        self.camera_matrices = self.compute_camera_matrices(self.camera_position, self.camera_look_at, self.camera_up)
        self.perspective_transforms = self.camera_perspective(self.image_width / self.image_height, self.fov_y, self.near_clip, self.far_clip)
        self.clip_space_transforms = tf.matmul(self.perspective_transforms, self.camera_matrices)

        # --------------------------------------------------------------------------------------------------------------
        self.background_value = [-1] * self.vertex_attributes.shape[2].value

        # rasterize triangles
        self.pixel_attributes = self.rasterize_triangles(self.vertices,
                                                         self.vertex_attributes,
                                                         self.triangles,
                                                         self.clip_space_transforms,
                                                         self.background_value)

        # Extract the interpolated vertex attributes from the pixel buffer and supply them to the shader:
        pixel_normals = tf.nn.l2_normalize(self.pixel_attributes[:, :, :, 0:3], axis=3)
        pixel_positions = self.pixel_attributes[:, :, :, 3:6]
        diffuse_colors = self.pixel_attributes[:, :, :, 6:9]

        if specular_colors is not None:
            specular_colors = self.pixel_attributes[:, :, :, 9:12]
            # Retrieve the interpolated shininess coefficients if necessary, or just
            # reshape our input for broadcasting:
            if len(shininess_coefficients.shape) == 2:
                shininess_coefficients = self.pixel_attributes[:, :, :, 12]
            else:
                shininess_coefficients = tf.reshape(shininess_coefficients, [-1, 1, 1])

        pixel_mask = tf.cast(tf.reduce_any(diffuse_colors >= 0, axis=3), tf.float32)

        self.renderer = self.phong_shader(normals=pixel_normals,
                                          alphas=pixel_mask,
                                          pixel_positions=pixel_positions,
                                          light_positions=light_positions,
                                          light_intensities=light_intensities,
                                          diffuse_colors=diffuse_colors,
                                          camera_position=camera_position if specular_colors is not None else None,
                                          specular_colors=specular_colors,
                                          shininess_coefficients=shininess_coefficients,
                                          ambient_color=self.ambient_color)

    def run(self, inputs):
        session = tfSession()
        return session.run(inputs)

    # ------------------------------------------------------------------------------------------------------------------
    def phong_shader(self,
                     normals,
                     alphas,
                     pixel_positions,
                     light_positions,
                     light_intensities,
                     diffuse_colors=None,
                     camera_position=None,
                     specular_colors=None,
                     shininess_coefficients=None,
                     ambient_color=None):
        """Computes pixelwise lighting from rasterized buffers with the Phong model.

        Args:
          normals: a 4D float32 tensor with shape [batch_size, image_height,
              image_width, 3]. The inner dimension is the world space XYZ normal for
              the corresponding pixel. Should be already normalized.
          alphas: a 3D float32 tensor with shape [batch_size, image_height,
              image_width]. The inner dimension is the alpha value (transparency)
              for the corresponding pixel.
          pixel_positions: a 4D float32 tensor with shape [batch_size, image_height,
              image_width, 3]. The inner dimension is the world space XYZ position for
              the corresponding pixel.
          light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
              XYZ position of each light in the scene. In the same coordinate space as
              pixel_positions.
          light_intensities: a 3D tensor with shape [batch_size, light_count, 3]. The
              RGB intensity values for each light. Intensities may be above one.
          diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
              image_width, 3]. The inner dimension is the diffuse RGB coefficients at
              a pixel in the range [0, 1].
          camera_position: a 1D tensor with shape [batch_size, 3]. The XYZ camera
              position in the scene. If supplied, specular reflections will be
              computed. If not supplied, specular_colors and shininess_coefficients
              are expected to be None. In the same coordinate space as
              pixel_positions.
          specular_colors: a 4D float32 tensor with shape [batch_size, image_height,
              image_width, 3]. The inner dimension is the specular RGB coefficients at
              a pixel in the range [0, 1]. If None, assumed to be tf.zeros()
          shininess_coefficients: A 3D float32 tensor that is broadcasted to shape
              [batch_size, image_height, image_width]. The inner dimension is the
              shininess coefficient for the object at a pixel. Dimensions that are
              constant can be given length 1, so [batch_size, 1, 1] and [1, 1, 1] are
              also valid input shapes.
          ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
              color, which is added to each pixel before tone mapping. If None, it is
              assumed to be tf.zeros().
        Returns:
          A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
          containing the lit RGBA color values for each image at each pixel. Colors
          are in the range [0,1].

        Raises:
          ValueError: An invalid argument to the method is detected.
        """
        batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]
        light_count = light_positions.shape[1].value
        pixel_count = image_height * image_width

        # Reshape all values to easily do pixelwise computations:
        normals = tf.reshape(normals, [batch_size, -1, 3])
        alphas = tf.reshape(alphas, [batch_size, -1, 1])
        diffuse_colors = tf.reshape(diffuse_colors, [batch_size, -1, 3])

        if camera_position is not None:
            specular_colors = tf.reshape(specular_colors, [batch_size, -1, 3])

        # Ambient component
        output_colors = tf.zeros([batch_size, image_height * image_width, 3])
        if ambient_color is not None:
            ambient_reshaped = tf.expand_dims(ambient_color, axis=1)
            output_colors = tf.add(output_colors, ambient_reshaped * diffuse_colors)

        # Diffuse component
        pixel_positions = tf.reshape(pixel_positions, [batch_size, -1, 3])
        per_light_pixel_positions = tf.stack(
            [pixel_positions] * light_count,
            axis=1)  # [batch_size, light_count, pixel_count, 3]

        directions_to_lights = tf.nn.l2_normalize(
            tf.expand_dims(light_positions, axis=2) - per_light_pixel_positions,
            axis=3)  # [batch_size, light_count, pixel_count, 3]

        # The specular component should only contribute when the light and normal
        # face one another (i.e. the dot product is nonnegative):
        normals_dot_lights = tf.clip_by_value(
            tf.reduce_sum(
                tf.expand_dims(normals, axis=1) * directions_to_lights, axis=3), 0.0,
            1.0)  # [batch_size, light_count, pixel_count]

        diffuse_output = tf.expand_dims(
            diffuse_colors, axis=1) * tf.expand_dims(
            normals_dot_lights, axis=3) * tf.expand_dims(
            light_intensities, axis=2)

        diffuse_output = tf.reduce_sum(diffuse_output, axis=1)  # [batch_size, pixel_count, 3]
        output_colors = tf.add(output_colors, diffuse_output)

        # Specular component
        if camera_position is not None:
            camera_position = tf.reshape(camera_position, [batch_size, 1, 3])
            mirror_reflection_direction = tf.nn.l2_normalize(
                2.0 * tf.expand_dims(normals_dot_lights, axis=3) * tf.expand_dims(
                    normals, axis=1) - directions_to_lights,
                dim=3)
            direction_to_camera = tf.nn.l2_normalize(
                camera_position - pixel_positions, dim=2)
            reflection_direction_dot_camera_direction = tf.reduce_sum(
                tf.expand_dims(direction_to_camera, axis=1) *
                mirror_reflection_direction,
                axis=3)

            # The specular component should only contribute when the reflection is
            # external:
            reflection_direction_dot_camera_direction = tf.clip_by_value(
                tf.nn.l2_normalize(reflection_direction_dot_camera_direction, dim=2),
                0.0, 1.0)
            # The specular component should also only contribute when the diffuse
            # component contributes:
            reflection_direction_dot_camera_direction = tf.where(
                normals_dot_lights != 0.0, reflection_direction_dot_camera_direction,
                tf.zeros_like(
                    reflection_direction_dot_camera_direction, dtype=tf.float32))

            # Reshape to support broadcasting the shininess coefficient, which rarely
            # varies per-vertex:
            reflection_direction_dot_camera_direction = tf.reshape(reflection_direction_dot_camera_direction,
                                                                   [batch_size, light_count, image_height, image_width])
            shininess_coefficients = tf.expand_dims(shininess_coefficients, axis=1)
            specularity = tf.reshape(tf.pow(reflection_direction_dot_camera_direction, shininess_coefficients),
                                     [batch_size, light_count, pixel_count, 1])

            specular_output = tf.expand_dims(
                specular_colors, axis=1) * specularity * tf.expand_dims(
                light_intensities, axis=2)
            specular_output = tf.reduce_sum(specular_output, axis=1)
            output_colors = tf.add(output_colors, specular_output)

        rgb_images = tf.reshape(output_colors, [batch_size, image_height, image_width, 3])
        alpha_images = tf.reshape(alphas, [batch_size, image_height, image_width, 1])
        valid_rgb_values = tf.concat(3 * [alpha_images > 0.5], axis=3)

        rgb_images = tf.where(valid_rgb_values, rgb_images, tf.zeros_like(rgb_images, dtype=tf.float32))

        return tf.reverse(tf.concat([rgb_images, alpha_images], axis=3), axis=[1])

    # ------------------------------------------------------------------------------------------------------------------
    def rasterize_triangles(self,
                            vertices,
                            attributes,
                            triangles,
                            projection_matrices,
                            background_value):

        """Rasterizes the input scene and computes interpolated vertex attributes.

        NOTE: the rasterizer does no triangle clipping. Triangles that lie outside the
        viewing frustum (esp. behind the camera) may be drawn incorrectly.

        Args:
          vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
              triplet is an xyz position in model space.
          attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
              attribute_count]. Each vertex attribute is interpolated
              across the triangle using barycentric interpolation.
          triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
              should contain vertex indices describing a triangle such that the
              triangle's normal points toward the viewer if the forward order of the
              triplet defines a clockwise winding of the vertices. Gradients with
              respect to this tensor are not available.
          projection_matrices: 3-D float tensor with shape [batch_size, 4, 4]
              containing model-view-perspective projection matrices.
          background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
              that lie outside all triangles take this value.

        Returns:
          A 4-D float32 tensor with shape [batch_size, image_height, image_width,
          attribute_count], containing the interpolated vertex attributes at
          each pixel.

        Raises:
          ValueError: An invalid argument to the method is detected.
        """

        batch_size = vertices.shape[0].value
        vertex_count = vertices.shape[1].value

        # We map the coordinates to normalized device coordinates before passing
        # the scene to the rendering kernel to keep as many ops in tensorflow as possible.
        homogeneous_coord = tf.ones([batch_size, vertex_count, 1], dtype=tf.float32)
        vertices_homogeneous = tf.concat([vertices, homogeneous_coord], 2)

        # Vertices are given in row-major order, but the transformation pipeline is column major:
        clip_space_points = tf.matmul(vertices_homogeneous, projection_matrices, transpose_b=True)

        # perspective divide, first thresholding the homogeneous coordinate to avoid the possibility of NaNs:
        clip_space_points_w = tf.maximum(tf.abs(clip_space_points[:, :, 3:4]),
                                         minimum_perspective_threshold) * tf.sign(clip_space_points[:, :, 3:4])
        self.normalized_device_coordinates = clip_space_points[:, :, 0:3] / clip_space_points_w

        x_image_coordinates = (self.normalized_device_coordinates[:, :, 0] + 1) * self.image_width/2
        x_image_coordinates = tf.expand_dims(x_image_coordinates, axis=1)

        y_image_coordinates = (1 - self.normalized_device_coordinates[:, :, 1]) * self.image_height/2
        y_image_coordinates = tf.expand_dims(y_image_coordinates, axis=1)

        self.image_coordinates = tf.concat((x_image_coordinates, y_image_coordinates), axis=1)

        self.per_image_vertex_ids = []
        self.per_image_uncorrected_barycentric_coordinates = []

        for im in range(vertices.shape[0]):
            barycentric_coords, triangle_ids, _ = \
                rasterize_triangles_module.rasterize_triangles(
                    self.normalized_device_coordinates[im], triangles, self.image_width, self.image_height)

            self.per_image_uncorrected_barycentric_coordinates.append(tf.reshape(barycentric_coords, [-1, 3]))

            # Gathers the vertex indices now because the indices don't contain a batch identifier,
            # and reindexes the vertex ids to point to a (batch,vertex_id)
            vertex_ids_ = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
            self.per_image_vertex_ids.append(tf.add(vertex_ids_, im * vertices.shape[1].value))

        self.uncorrected_barycentric_coordinates = tf.concat(self.per_image_uncorrected_barycentric_coordinates, axis=0)
        self.vertex_ids = tf.concat(self.per_image_vertex_ids, axis=0)

        # Indexes with each pixel's clip-space triangle's extrema (the pixel's 'corner points') ids
        # to get the relevant properties for deferred shading.
        flattened_vertex_attributes = tf.reshape(attributes, [batch_size * vertex_count, -1])
        corner_attributes = tf.gather(flattened_vertex_attributes, self.vertex_ids)

        # Barycentric interpolation is linear in the reciprocal of the homogeneous
        # W coordinate, so we use these weights to correct for the effects of
        # perspective distortion after rasterization.
        perspective_distortion_weights = tf.reciprocal(tf.reshape(clip_space_points_w, [-1]))
        corner_distortion_weights = tf.gather(perspective_distortion_weights, self.vertex_ids)

        # Apply perspective correction to the barycentric coordinates. This step is
        # required since the rasterizer receives normalized-device coordinates (i.e.,
        # after perspective division), so it can't apply perspective correction to the interpolated values.
        weighted_barycentric_coordinates = tf.multiply(self.uncorrected_barycentric_coordinates, corner_distortion_weights)
        barycentric_reweighting_factor = tf.reduce_sum(weighted_barycentric_coordinates, axis=1)

        corrected_barycentric_coordinates = \
            tf.divide(weighted_barycentric_coordinates,
                  tf.expand_dims(tf.maximum(barycentric_reweighting_factor, minimum_reweighting_threshold), axis=1))

        # Computes the pixel attributes by interpolating the known attributes at the
        # corner points of the triangle interpolated with the barycentric coordinates.
        weighted_vertex_attributes = tf.multiply(corner_attributes,
                                                 tf.expand_dims(corrected_barycentric_coordinates, axis=2))
        summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
        attribute_images = tf.reshape(summed_attributes, [batch_size, self.image_height, self.image_width, -1])

        # Barycentric coordinates should approximately sum to one where there is
        # rendered geometry, but be exactly zero where there is not.
        alphas = tf.clip_by_value(tf.reduce_sum(2.0 * corrected_barycentric_coordinates, axis=1), 0.0, 1.0)
        alphas = tf.reshape(alphas, [batch_size, self.image_height, self.image_width, 1])

        attributes_with_background = alphas * attribute_images + (1.0 - alphas) * background_value

        return attributes_with_background

    # ------------------------------------------------------------------------------------------------------------------
    def compute_camera_matrices(self, camera_position, camera_look_at, camera_up):
        """Computes camera viewing matrices.

        Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

        Args:
          camera_position: 2-D float32 tensor with shape [batch_size, 3] containing the XYZ world
              space position of the camera.
          camera_look_at: 2-D float32 tensor with shape [batch_size, 3] containing a position
              along the center of the camera's gaze.
          camera_up: 2-D float32 tensor with shape [batch_size, 3] specifying the
              world's up direction; the output camera will have no tilt with respect
              to this direction.

        Returns:
          A [batch_size, 4, 4] float tensor containing a right-handed camera
          extrinsics matrix that maps points from world space to points in eye space.
        """
        batch_size = camera_look_at.shape[0].value
        vector_degeneracy_cutoff = 1e-6

        forward = camera_look_at - camera_position
        forward_norm = tf.norm(forward, ord='euclidean', axis=1, keepdims=True)
        tf.assert_greater(forward_norm, vector_degeneracy_cutoff,
                          message='Camera matrix is degenerate because eye and center are close.')
        forward = tf.divide(forward, forward_norm)

        to_side = tf.cross(forward, camera_up)
        to_side_norm = tf.norm(to_side, ord='euclidean', axis=1, keepdims=True)
        tf.assert_greater(to_side_norm, vector_degeneracy_cutoff,
                          message='Camera matrix is degenerate because up and gaze are close or because up is degenerate.')

        to_side = tf.divide(to_side, to_side_norm)
        cam_up = tf.cross(to_side, forward)

        w_column = tf.constant(batch_size * [[0., 0., 0., 1.]], dtype=tf.float32)
        w_column = tf.reshape(w_column, [batch_size, 4, 1])
        view_rotation = tf.stack([to_side, cam_up, -forward, tf.zeros_like(to_side, dtype=tf.float32)], axis=1)
        view_rotation = tf.concat([view_rotation, w_column], axis=2)

        identity_batch = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
        view_translation = tf.concat([identity_batch, tf.expand_dims(-camera_position, 2)], 2)
        view_translation = tf.concat([view_translation, tf.reshape(w_column, [batch_size, 1, 4])], 1)
        camera_matrices = tf.matmul(view_rotation, view_translation)

        return camera_matrices

    # ------------------------------------------------------------------------------------------------------------------
    def camera_perspective(self, aspect_ratio, fov_y, near_clip, far_clip):
        """Computes perspective transformation matrices.

        Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).

        Args:
          aspect_ratio: float value specifying the image aspect ratio (width/height).
          fov_y: 1-D float32 Tensor with shape [batch_size] specifying output vertical
              field of views in degrees.
          near_clip: 1-D float32 Tensor with shape [batch_size] specifying near
              clipping plane distance.
          far_clip: 1-D float32 Tensor with shape [batch_size] specifying far clipping
              plane distance.

        Returns:
          A [batch_size, 4, 4] float tensor that maps from right-handed points in eye
          space to left-handed points in clip space.
        """
        # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
        # and adds the half-angle factor of .5.
        focal_lengths_y = 1.0 / tf.tan(fov_y * (math.pi / 360.0))
        depth_range = far_clip - near_clip
        p_22 = -(far_clip + near_clip) / depth_range
        p_23 = -2.0 * (far_clip * near_clip / depth_range)

        zeros = tf.zeros_like(p_23, dtype=tf.float32)
        perspective_transform = tf.concat(
            [
                focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
                zeros, focal_lengths_y, zeros, zeros,
                zeros, zeros, p_22, p_23,
                zeros, zeros, -tf.ones_like(p_23, dtype=tf.float32), zeros
            ], axis=0)
        perspective_transform = tf.reshape(perspective_transform, [4, 4, -1])

        return tf.transpose(perspective_transform, [2, 0, 1])

    # @tf.RegisterGradient('RasterizeTriangles')
    # def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
    #     # Gradients are only supported for barycentric coordinates. Gradients for the
    #     # z-buffer are possible as well but not currently implemented.
    #     del df_dids, df_dz
    #     return rasterize_triangles_module.rasterize_triangles_grad(
    #         op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
    #         op.get_attr('image_width'), op.get_attr('image_height')), None
