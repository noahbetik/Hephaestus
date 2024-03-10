# line sketching demo

import open3d as o3d
import numpy as np
import time
from open3d.utility import Vector3dVector
from open3d.utility import Vector2iVector

'''n_new = 2

# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
# *optionally* add initial points
points = np.random.rand(n_new, 3)
lines = np.array([[i, i+1] for i in range(n_new-1)])
pcd.points = o3d.utility.Vector3dVector(points)

ls = o3d.geometry.LineSet()
ls.points = Vector3dVector(points)
ls.lines = Vector2iVector(lines)


# include it in the visualizer before non-blocking visualization.
vis.add_geometry(pcd)
vis.add_geometry(ls)

# to add new points each dt secs.
dt = 0.01
# number of points that will be added


previous_t = time.time()

# run non-blocking visualization. 
# To exit, press 'q' or click the 'x' of the window.
keep_running = True
while keep_running:
    
    if time.time() - previous_t > dt:
        # Options (uncomment each to try them out):
        # 1) extend with ndarrays.
        add_this = np.random.rand(n_new, 3)
        pcd.points.extend(add_this)
        ls.points.extend(add_this)
        add_ls = np.array([[i, i+1] for i in range(len(pcd.points)-n_new, len(pcd.points))])
        ls.lines.extend(Vector2iVector(add_ls))
        
        # 2) extend with Vector3dVector instances.
        # pcd.points.extend(
        #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
        
        # 3) other iterables, e.g
        # pcd.points.extend(np.random.rand(n_new, 3).tolist())
        
        vis.update_geometry(pcd)
        vis.update_geometry(ls)
        previous_t = time.time()

    keep_running = vis.poll_events()
    vis.update_renderer()
'''



# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

points = np.array([[0, 0, 0], [1, 0, 0], [2, 1.5, 0], [1, 3, 0], [0, 3, 0], [-1, 1.5, 0]])
lines = np.array([[0, 1], [1,2], [2,3], [3,4], [4,5]])

pcd = o3d.geometry.PointCloud()
pcd.points = Vector3dVector(points)

ls = o3d.geometry.LineSet()
ls.points = Vector3dVector(points)
ls.lines = Vector2iVector(lines)

vis.add_geometry(pcd)
vis.add_geometry(ls)




while "kor":
        
    keep_running = vis.poll_events()
    vis.update_renderer()
        

