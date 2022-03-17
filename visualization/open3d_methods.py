import numpy as np
import open3d as o3d

def PointCloudVisualize(points, colors):
    points,colors
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


def DepthFilter(points_3d,colors,depth):
    flag = (depth>0) & (depth<45)
    real_points_3d = points_3d[flag,:]
    real_colors = colors[flag,:]
    return real_points_3d, real_colors
