# custom point cloud demo

import open3d as o3d
import numpy as np


points1 = np.array([[1., 1., 1.], [2., 1., 1.], [2., 2., 1.], [1., 2., 1.],
           [1., 1., 2.], [2., 1., 2.], [2., 2., 2.], [1., 2., 2.]])

lines1 = np.array([[0, 1], [1, 2], [2, 3], [3, 0], \
          [4, 5], [5, 6], [6, 7], [7, 4], \
          [0, 4], [1, 5], [2, 6], [3, 7]])

points2 = []
lines2 = lines1

for coord in points1:
    temp = [coord[0]+2, coord[1]+2, coord[2]+2]
    points2.append(temp)

pcd1 = o3d.geometry.PointCloud()
print(pcd1, "\n")
pcd1.points = o3d.utility.Vector3dVector(points1)
pcd1.estimate_normals()

'''colors1 = [[1, 0, 0] for i in range(len(lines1))]
ls1 = o3d.geometry.LineSet()

colors2 = [[1, 0, 0] for i in range(len(lines1))]
ls2 = o3d.geometry.LineSet()

ls1.points = o3d.utility.Vector3dVector(points1)
ls1.lines = o3d.utility.Vector2iVector(lines1)
ls1.colors = o3d.utility.Vector3dVector(colors1)

ls2.points = o3d.utility.Vector3dVector(points2)
ls2.lines = o3d.utility.Vector2iVector(lines2)
ls2.colors = o3d.utility.Vector3dVector(colors2)'''

mesh1, densities1 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd1, depth=9)

mesh2 = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
mesh2.compute_vertex_normals()
#mesh2.translate(np.array([1, 1, 0]))

#mesh3 = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
#mesh3.compute_vertex_normals()
#mesh3.translate(np.array([1, 3, 0]))

#mesh1 += ls1
print(mesh1)

print(type(mesh1))
meshes = [mesh2]

o3d.visualization.draw_geometries(meshes)

i = 1

while True:
    repeat = input()
    if repeat == "k":
        newBox = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        newBox.compute_vertex_normals()
        newBox.translate(np.array([i, i, i]))
        meshes.append(newBox)
        i += 1
        o3d.visualization.draw_geometries(meshes)
