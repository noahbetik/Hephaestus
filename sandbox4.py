import numpy as np
import open3d as o3d

# Generate uniformly distributed points in a cube
points = np.random.rand(1000, 3)

# Create a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Estimate normals for the point cloud
pcd.estimate_normals()

# Create a mesh from the point cloud using Poisson surface reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

# Visualize the mesh
o3d.visualization.draw_geometries([pcd, mesh])
