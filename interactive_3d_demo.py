import open3d as o3d
import numpy as np
import time

camera_parameters = o3d.camera.PinholeCameraParameters()

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

# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Open3D', width=width, height=height, left=50, top=50, visible=True)

ctr = vis.get_view_control()

# to add new points each dt secs.
dt = 1

tPrev = time.time()

# run non-blocking visualization. 
# To exit, press 'q' or click the 'x' of the window.
keepRunning = True
i = 0

while keepRunning:
    
    if time.time() - tPrev > dt:

        # Store the current view matrix
        current_view_matrix = ctr.convert_to_pinhole_camera_parameters().extrinsic

        newBox = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        newBox.compute_vertex_normals()
        newBox.translate(np.array([i, i, i]))
        vis.add_geometry(newBox)
        i += 1
        
        # Set back the stored view matrix
        camera_parameters.extrinsic = current_view_matrix
        ctr.convert_from_pinhole_camera_parameters(camera_parameters, True)

        tPrev = time.time()

    keepRunning = vis.poll_events()
    
    vis.update_renderer()

vis.destroy_window()
