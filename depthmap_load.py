import cv2
import glob
import time
import numpy as np
from epipolar.opencv_methods import EpipolarRecitification
from depthmap_sgbm.opencv_methods import StereoDepthEstimation
from visualization.open3d_methods import Open3DVisualizer


def NormPointCloud(points):
    x, y, z = np.array_split(points, 3, axis=1)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    x_norm = x - x_mean
    y_norm = y - y_mean
    z_norm = z_mean - z
    outpoints = np.stack([x_norm, y_norm, z_norm], axis=1)[:, :, 0]
    return outpoints


def DepthFilter(points, colors, threshold_low: int = 0, threshold_high: int = 45):
    """Remove useless points

    Args:
        points (np.ndarray): the positions of the points
        colors (np.ndarray): the depth of all points
        threshold_low (int, optional): the low threshold. Defaults to 0.
        threshold_high (int, optional): the high threshold. Defaults to 45.
    """
    depth = points[:, -1]
    flag = (depth > threshold_low) & (depth < threshold_high)
    points = points[flag, :]
    colors = colors[flag, :]
    return points, colors

def MaskDepthFilter(points, colors, mask):
    """Remove useless points

    Args:
        points (np.ndarray): the positions of the points
        colors (np.ndarray): the depth of all points
        mask (np.ndarray): the mask of useful points
    """
    points = points[mask, :]
    colors = colors[mask, :]
    return points, colors

class StereoReconstruction(object):
    """Stereo reconstruction function

    Args:
        imgshape (tuple): the shape of the image
        camera1_name (str): the name of the camera one
        camera2_name (str): the name of the camera two
        jsonpath (str): the filename of the viewpoint
    """

    def __init__(
        self,
        imgshape: tuple,
        camera1_name: str,
        camera2_name: str,
        jsonpath: str,
    ):
        """Stereo reconstruction function

        Args:
            imgshape (tuple): the shape of the image
            camera1_name (str): the name of the camera one
            camera2_name (str): the name of the camera two
            jsonpath (str): the filename of the viewpoint
        """
        self.stereoCamera = StereoDepthEstimation(imgshape, camera1_name, camera2_name)
        self.visualizer = Open3DVisualizer(jsonpath)

    def PointCloudGenerator(self, img_left: np.ndarray, img_right: np.ndarray):
        """Point cloud generation

        Args:
            img_left (np.ndarray): the image of the left camera
            img_right (np.ndarray): the image of the right camera

        Returns:
            points: the positions of the points
            colors: the depth of all points
        """
        _, result_left, result_right = EpipolarRecitification(
            self.stereoCamera.camera1_cfg,
            self.stereoCamera.camera2_cfg,
            img_left,
            img_right,
        )
        disparity = np.load("left.npy")[1:, :]
        disparity = cv2.bilateralFilter(disparity, 3, 30, 30)
        mask = disparity.reshape((-1))>-80
        points_3d, depth = self.stereoCamera.DepthEstimation(disparity)

        # from post_process.post_process import get_processed

        # points_3d = get_processed(points_3d, result_left, iterations=3)
        # cv2.imwrite("output/sgbm/depthmap_" + str(i) + ".png", disp)
        # cv2.imwrite("output/sgbm/depthmap_bf_" + str(i) + ".png", disp_bf)
        points = points_3d.reshape((-1, 3))
        colors = result_left.reshape((-1, 3)) / 256.0
        points, colors = MaskDepthFilter(points, colors, mask)
        return points, colors

    def PointCloudVisualization(self, points, colors):
        """Point cloud visualization

        Args:
            points (np.ndarray): the positions of the points
            colors (np.ndarray): the depth of all points
        """
        self.visualizer.UpdatePointCloud(points, colors)


if __name__ == "__main__":
    imgshape = (1280, 720)
    camera1_name = "left"
    camera2_name = "right"
    jsonpath = "output/config/visualization/view_point.json"
    stereoCamera = StereoReconstruction(imgshape, camera1_name, camera2_name, jsonpath)

    test_left = glob.glob("image/test/left/*.png")
    test_right = glob.glob("image/test/right/*.png")

    for i in range(len(test_left)):
        img_left = cv2.imread(test_left[i])
        img_right = cv2.imread(test_right[i])
        points, colors = stereoCamera.PointCloudGenerator(img_left, img_right)
        # points = np.load("points.npy")
        # colors = np.load("colors.npy")

        points = NormPointCloud(points)

        points, colors = DepthFilter(
            points, colors, threshold_low=-500, threshold_high=500
        )

        stereoCamera.PointCloudVisualization(points, colors)
        # time.sleep(20)
        # stereoCamera.visualizer.SaveViewPoint(jsonpath)
