# open3d demo

import open3d as o3d
import numpy as np

print("Testing mesh in Open3D...")
armadillo_mesh = o3d.data.ArmadilloMesh()
mesh1 = o3d.io.read_triangle_mesh(armadillo_mesh.path)
print(mesh1)
print('Vertices:')
print(np.asarray(mesh1.vertices))
print('Triangles:')
print(np.asarray(mesh1.triangles))


knot_mesh = o3d.data.KnotMesh()
mesh2 = o3d.io.read_triangle_mesh(knot_mesh.path)
print(mesh2)
print('Vertices:')
print(np.asarray(mesh2.vertices))
print('Triangles:')
print(np.asarray(mesh2.triangles))


print("Computing normal and rendering it.")
mesh1.compute_vertex_normals()
mesh2.compute_vertex_normals()

print(np.asarray(mesh1.triangle_normals))
print(np.asarray(mesh2.triangle_normals))

o3d.visualization.draw_geometries([mesh1, mesh2])

