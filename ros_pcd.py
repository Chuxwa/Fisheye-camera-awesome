#!/usr/bin/python3
import rospy
import struct
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import cv2
import numpy as np
from depthmap_raft.raft import RAFTDepthEstimation
import argparse
from cv_bridge import CvBridge
import time
import open3d as o3d


def MaskDepthFilter(points: np.ndarray, colors: np.ndarray, mask: np.ndarray):
    """Remove useless points

    Args:s
        points (np.ndarray): the positions of the points
        colors (np.ndarray): the depth of all points
        mask (np.ndarray): the mask of useful points
    """
    points = points[mask, :]
    colors = colors[mask, :]
    return points, colors


def CropImage(img: np.ndarray, height: int = 720, width: int = 1280):
    """Crop image

    Args:
        img (np.ndarray): the source image
        height (int, optional): the height of the cropped image. Defaults to 720.
        width (int, optional): the width of the cropped image. Defaults to 1280.

    Returns:
        img: cropped image
    """
    src_height, src_width = img.shape[0], img.shape[1]
    start_width = (src_width - width) // 2
    start_height = src_height - height
    cropped_img = img[
        start_height : start_height + height, start_width : start_width + width, :
    ]
    return cropped_img


def ResizeImage(img: np.ndarray, depth: np.ndarray):
    """Resize depth map to the same size of image

    Args:
        img (np.ndarray): the source image
        depth (np.ndarray): the depth map

    Returns:
        depth: the depth map
    """
    height, width = img.shape[0], img.shape[1]
    src_height, src_width = depth.shape[0], depth.shape[1]
    start_width = (src_width - width) // 2
    start_height = src_height - height
    cropped_depth = depth[
        start_height : start_height + height, start_width : start_width + width
    ]
    # depth = cv2.resize(depth, (height, width))
    return cropped_depth

def DisplayInlier(
    pointcloud,
    voxel_size: float = 0.2,
    nb_points: int = 16,
    radius: float = 6.0,
):
    """Remove outlier points

    Args:
        voxel_size (float, optional): voxelization. Defaults to 0.2.
        nb_points (int, optional): neighborhood points. Defaults to 16.
        radius (float, optional): radius. Defaults to 6.0.
    """
    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)
    _, ind = pointcloud.remove_radius_outlier(
        nb_points=nb_points, radius=radius
    )
    pointcloud = pointcloud.select_by_index(ind)
    return pointcloud


def DepthEstimation(unnorm_disp, Q, scale: float = 1.0):
    """Reproject Image To 3D points

    Args:
        unnorm_disp (ndarray): the unnormlized disparity map
        scale (float): the scale factor of the depth
    Returns:
        points_3d (ndarray): the positions of the points
        depth (ndarray): the depth of all points
    """
    points_3d = cv2.reprojectImageTo3D(unnorm_disp, Q)
    _, _, depth = cv2.split(points_3d)
    depth = depth * scale
    depth = np.asarray(depth, dtype=np.float32)
    return points_3d, depth


class StereoReconstruction(object):
    """Stereo reconstruction function"""

    def __init__(self, args):
        self.depthestimation = RAFTDepthEstimation(args)
        self.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.UINT32, 1),
        ]
        camera_front_left = message_filters.Subscriber(
            "/miivii_gmsl_1/cam_front_left/image_rect_color", Image
        )
        camera_front_right = message_filters.Subscriber(
            "/miivii_gmsl_1/cam_front_right/image_rect_color", Image
        )
        self.cv_bridge = CvBridge()
        sync = message_filters.ApproximateTimeSynchronizer(
            [camera_front_left, camera_front_right], 10, 1
        )
        sync.registerCallback(self.multi_callback)
        self.pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)
        self.header = Header()
        self.header.frame_id = "cam_front_left_link"
        self.Q = np.array([
            [1.0,0.0,0.0,-675.4726488872279],
            [0.0,1.0,0.0,-351.03471467794805],
            [0.0,0.0,0.0,217.58312640799295],
            [0.0,0.0,0.15589079381539744,0.0]
        ])

    def multi_callback(self, img_left, img_right):
        """Point cloud generation

        Args:
            img_left (np.ndarray): the image of the left camera
            img_right (np.ndarray): the image of the right camera

        Returns:
            points: the positions of the points
            colors: the depth of all points
        """
        start = time.time()
        self.header.stamp = img_left.header.stamp
        img_left = self.cv_bridge.imgmsg_to_cv2(img_left, "bgr8")
        img_right = self.cv_bridge.imgmsg_to_cv2(img_right, "bgr8")
        result_left = CropImage(img_left, 600, 1280)
        result_right = CropImage(img_right, 600, 1280)
        result_left = cv2.bilateralFilter(result_left, 5, 15, 15)
        result_right = cv2.bilateralFilter(result_right, 5, 15, 15)
        disparity = self.depthestimation.run(result_left, result_right)
        disparity = ResizeImage(result_left, disparity)
        points_3d, depth = DepthEstimation(disparity, self.Q)
        mask = depth.reshape((-1)) < 500
        points = points_3d.reshape((-1, 3))
        colors = result_left.reshape((-1, 3))

        mask = (
            (points[:, 0] > -300)
            & (points[:, 0] < 200)
            & (points[:, 1] > -200)
            & (points[:, 1] < 55)
            & (points[:, 2] < 300)
            & (points[:, 2] > 50)
        )
        points, colors = MaskDepthFilter(points, colors, mask)
        # end = time.time()
        # print(end-start)
        # pointcloud = o3d.geometry.PoinstCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(points)
        # pointcloud.colors = o3d.utility.Vector3dVector(colors)
        # pointcloud = DisplayInlier(pointcloud, nb_points = 16, radius = 5)
        # points = np.asarray(pointcloud.points)
        # colors = np.asarray(pointcloud.colors)
        a = np.array([[255],]* colors.shape[0])
        rgba = np.hstack((colors, a)).reshape((-1)).astype(int)
        rgb = struct.unpack('I'* colors.shape[0], struct.pack('BBBB'* colors.shape[0], *rgba))
        out = np.hstack((points, np.array(rgb).reshape((-1,1)))).astype(int).tolist()
        pc2 = point_cloud2.create_cloud(self.header, self.fields, out)
        pc2.header.stamp = rospy.Time.now()
        self.pub.publish(pc2)
        end = time.time()
        print(end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.restore_ckpt = "image/ckpt/raftstereo-realtime.pth"
    args.mixed_precision = True
    args.valid_iters = 7
    args.hidden_dims = [128] * 3
    args.corr_implementation = "reg_cuda"
    args.corr_levels = 4
    args.corr_radius = 4
    args.n_downsample = 3
    args.n_gru_layers = 2
    args.shared_backbone = True
    args.slow_fast_gru = True
    rospy.init_node("ros_create_cloud_xyzrgb")
    # rospy.init_node("create_cloud_xyzrgb")
    pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)
    tensor = StereoReconstruction(args)
    rospy.spin()