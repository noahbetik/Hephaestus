from open3d import *
import open3d as o3d
import numpy as np
#import transforms3d
#from scipy.spatial.transform import *
#import matplotlib.pyplot as plt

# Set camera params.
camera_parameters = camera.PinholeCameraParameters()

width = 640
height = 480
focal = 0.9616278814278851

K = [[focal * width, 0, width / 2],
     [0, focal * width, height / 2],
     [0, 0, 1]]

camera_parameters.extrinsic = np.array([[1.204026265313826, 0, 0, -0.3759973645485034],
                                           [0, -0.397051999192357, 0,
                                               4.813624436153903, ],
                                           [0, 0, 0.5367143925232766,
                                               7.872266818189111],
                                           [0, 0, 0, 1]])

camera_parameters.intrinsic.set_intrinsics(
    width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])

# Visualize the scene.
viewer = visualization.Visualizer()
viewer.create_window(window_name='Open3D', width=width, height=width, left=50, top=50, visible=True)

control = viewer.get_view_control()
control.convert_from_pinhole_camera_parameters(camera_parameters, True)

newBox = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
newBox.compute_vertex_normals()
newBox.translate(np.array([1, 1, 1]))
viewer.add_geometry(newBox)
viewer.run()
