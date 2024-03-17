import random
import numpy as np
import open3d as o3d
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point

ps = [[0,0,0], [1,0,0], [1.5,1,0], [1,2,0], [0,2,0], [-0.5,1,0]]
ps2 = [[0,0,1], [1,0,1], [1.5,1,1], [1,2,1], [0,2,1], [-0.5,1,1]]


def generate_points_in_polygon(polygon, num_points):
    # Generate random points within the bounding box of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    print(min_x, min_y, max_x, max_y)
    points = []
    while len(points) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)
        if polygon.contains(point):
            points.append([x, y])

    # Create a Voronoi diagram from the points
    vor = Voronoi(points)

    # Calculate the centroid of each Voronoi cell
    centroids = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            vertices = [vor.vertices[i] for i in region]
            polygon = Polygon(vertices)
            centroid = polygon.centroid
            if (min_x <= centroid.x <= max_x and min_y <= centroid.y <= max_y):
                if (polygon.contains(Point(centroid.x, centroid.y))):
                    centroids.append([centroid.x, centroid.y, 0])

    return np.array(centroids)

def linear_interpolate_3d(point1, point2, n):
    # Calculate the step size for each dimension
    step = [(point2[i] - point1[i]) / (n + 1) for i in range(3)]

    # Generate the intermediate points
    intermediate_points = []
    for i in range(1, n + 1):
        x = point1[0] + step[0] * i
        y = point1[1] + step[1] * i
        z = point1[2] + step[2] * i
        intermediate_points.append([x, y, z])

    return intermediate_points

def scale_polygon_2d(polygon, scale_factor):
    # Compute the centroid of the polygon
    centroid = np.mean(polygon, axis=0)

    # Translate the polygon to the origin
    translated_polygon = polygon - centroid

    # Scale the polygon
    scaled_polygon = translated_polygon * scale_factor

    # Translate the scaled polygon back to its original position
    scaled_polygon += centroid

    print(scaled_polygon)

    return scaled_polygon.tolist()

# Define a polygon (in this case, a hexagon)
polygon = Polygon(ps)

# Generate points within the polygon
num_points = 1000
#points = generate_points_in_polygon(polygon, num_points)

points = ps

for p in range(len(ps) - 1):
    points = points + linear_interpolate_3d(ps[p], ps[p+1], 10)

points = points + linear_interpolate_3d(ps[-1], ps[0], 10)

# Create a point cloud from the generated points
pcd = o3d.geometry.PointCloud()

#ls = o3d.geometry.LineSet()
#ls.points = o3d.utility.Vector3dVector(np.array(ps))
#ls.lines = o3d.utility.Vector2iVector(np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]))

stacked = []

for p in points:
    for i in range(1, 10):
        stacked.append([p[0], p[1], i/10])

points = points + stacked

points = points + scale_polygon_2d(np.array(ps), 0.5)
points = points + scale_polygon_2d(np.array(ps2), 0.5)

pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals()


radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])
rec_mesh.compute_vertex_normals()

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd, rec_mesh])
