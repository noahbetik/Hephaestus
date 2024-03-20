import numpy as np
import open3d as o3d

# Generate some random points
np.random.seed(0)
points = np.random.rand(1000, 3)

# Create a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Estimate normals
pcd.estimate_normals()

# Reconstruct the surface using ball pivoting
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector([0.01, 0.01])
)

# Visualize the reconstructed mesh
o3d.visualization.draw_geometries([rec_mesh])
