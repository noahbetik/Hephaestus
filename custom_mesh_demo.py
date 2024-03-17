# line sketching demo

import open3d as o3d
import numpy as np
import time
from open3d.utility import Vector3dVector
from open3d.utility import Vector2iVector


# create visualizer and window.
#vis = o3d.visualization.Visualizer()
#vis.create_window(height=480, width=640)

# Parameters
radius = 1
height = 1
density = 100  # Number of points per unit length

# Calculate the number of points needed
num_points = int(np.ceil(2 * np.pi * radius * density))

# Generate points
points = []
for j in range(0,100):
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y, j/100])

min_x, max_x = -radius, radius
min_y, max_y = -radius, radius
step = 1 / density
for x in np.arange(min_x, max_x + step, step):
    for y in np.arange(min_y, max_y + step, step):
        if x**2 + y**2 <= radius**2:
            points.append([x, y, 0])

for x in np.arange(min_x, max_x + step, step):
    for y in np.arange(min_y, max_y + step, step):
        if x**2 + y**2 <= radius**2:
            points.append([x, y, 1])


# Convert to numpy array
points = np.array(points)

print("Number of points:", len(points))
print("Sample points:", points[:5])

# Convert to numpy array
points = np.array(points)

print("Number of points:", len(points))
print("Sample points:", points[:5])

pcd = o3d.geometry.PointCloud()
pcd.points = Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#vis.add_geometry(pcd)
#vis.add_geometry(ls)

# Visualize the point cloud to ensure it's correct
o3d.visualization.draw_geometries([pcd])
#o3d.visualization.draw_geometries([ls])

#eagle = o3d.data.EaglePointCloud()
#pcd = o3d.io.read_point_cloud(eagle.path)


with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

'''while True:
    keep_running = vis.poll_events()
    vis.update_renderer()'''
        

