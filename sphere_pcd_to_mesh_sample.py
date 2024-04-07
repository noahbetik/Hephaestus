import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

def fibonacci_sphere(samples=100, randomize=True):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = ((i + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])

    return points

# Generate points on the sphere
num_samples = 4000
sphere_points = fibonacci_sphere(samples=num_samples)


np_list = np.array(sphere_points)

# Create an Open3D point cloud from the generated points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_list)

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Optional: Orient the normals to point outwards
pcd.orient_normals_consistent_tangent_plane(k=30)

# Use ball pivoting algorithm to create a mesh with adjusted radii
# The radii should be chosen based on the distribution of your points
radii = [0.005, 0.01, 0.02, 0.04]  # Adjust these based on your specific point cloud
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

mesh_extrusion = o3d.t.geometry.TriangleMesh.from_legacy(rec_mesh)
for i in range(1, 31):
    mesh_extrusion.fill_holes(1 * (2 ** i))
filled = mesh_extrusion.to_legacy()

# Visualize the mesh and the point cloud
o3d.visualization.draw_geometries([filled])
