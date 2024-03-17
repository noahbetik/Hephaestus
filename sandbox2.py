import open3d as o3d
import numpy as np
import time
from open3d.utility import Vector3dVector
from open3d.utility import Vector2iVector


#bunny = o3d.data.BunnyMesh()
#mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
mesh.compute_vertex_normals()

pcd = mesh.sample_points_poisson_disk(3000)

o3d.visualization.draw_geometries([pcd])
'''alpha = 0.03
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)'''

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])

'''with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])'''
