import cv2
import glob
import time
import argparse
import numpy as np
from epipolar.opencv_methods import EpipolarRecitification
from depthmap_sgbm.opencv_methods import StereoDepthEstimation
from visualization.open3d_methods import Open3DVisualizer
from depthmap_raft.raft import RAFTDepthEstimation


def NormPointCloud(points):
    x, y, z = np.array_split(points, 3, axis=1)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    x_norm = x - x_mean
    y_norm = y - y_mean
    z_norm = z - z_mean
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


def MaskDepthFilter(points: np.ndarray, colors: np.ndarray, mask: np.ndarray):
    """Remove useless points

    Args:
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
    width, height = img.shape[0], img.shape[1]
    depth = cv2.resize(depth, (height, width))
    return depth


class StereoReconstruction(object):
    """Stereo reconstruction function

    Args:
        imgshape (tuple): the shape of the image
        camera1_name (str): the name of the camera one
        camera2_name (str): the name of the camera two
        jsonpath (str): the filename of the viewpoint
    """

    def __init__(self, args):
        """Stereo reconstruction function

        Args:
            imgshape (tuple): the shape of the image
            camera1_name (str): the name of the camera one
            camera2_name (str): the name of the camera two
            jsonpath (str): the filename of the viewpoint
        """
        self.stereoCamera = StereoDepthEstimation(
            args.imgshape, args.camera1_name, args.camera2_name
        )
        self.visualizer = Open3DVisualizer(args.jsonpath)
        self.depthestimation = RAFTDepthEstimation(args)

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
        result_left = CropImage(result_left, 600, 1280)
        result_right = CropImage(result_right, 600, 1280)

        result_left = cv2.bilateralFilter(result_left, 20, 75, 75)
        result_right = cv2.bilateralFilter(result_right, 20, 75, 75)
        disparity = self.depthestimation.run(result_left, result_right)
        disparity = ResizeImage(result_left, disparity)
        disparity = cv2.bilateralFilter(disparity, 3, 75, 75)
        cv2.imwrite("disparity.png", disparity)
        mask = disparity.reshape((-1)) > -80
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", help="restore checkpoint", required=True)
    parser.add_argument(
        "--imgshape", help="the shape of the image", default=(1280, 720)
    )
    parser.add_argument(
        "--camera1_name", help="the name of the first (left) camera", default="left"
    )
    parser.add_argument(
        "--camera2_name", help="the name of the second (right) camera", default="right"
    )
    parser.add_argument(
        "--jsonpath",
        help="the filename of the viewpoint",
        default="output/config/visualization/view_point.json",
    )
    parser.add_argument(
        "-l",
        "--left_imgs",
        help="path to all first (left) frames",
        default="image/test/left/*.png",
    )
    parser.add_argument(
        "-r",
        "--right_imgs",
        help="path to all second (right) frames",
        default="image/test/right/*.png",
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )

    # Architecture choices
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[128] * 3,
        help="hidden state and context dimensions",
    )
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
        help="correlation volume implementation",
    )
    parser.add_argument(
        "--shared_backbone",
        action="store_true",
        help="use a single backbone for the context and feature encoders",
    )
    parser.add_argument(
        "--corr_levels",
        type=int,
        default=4,
        help="number of levels in the correlation pyramid",
    )
    parser.add_argument(
        "--corr_radius", type=int, default=4, help="width of the correlation pyramid"
    )
    parser.add_argument(
        "--n_downsample",
        type=int,
        default=2,
        help="resolution of the disparity field (1/2^K)",
    )
    parser.add_argument(
        "--slow_fast_gru",
        action="store_true",
        help="iterate the low-res GRUs more frequently",
    )
    parser.add_argument(
        "--n_gru_layers", type=int, default=3, help="number of hidden GRU levels"
    )

    args = parser.parse_args()
    stereoCamera = StereoReconstruction(args)

    test_left = glob.glob(args.left_imgs)
    test_right = glob.glob(args.right_imgs)
    for i in range(len(test_left)):
        img_left = cv2.imread(test_left[i])
        img_right = cv2.imread(test_right[i])
        points, colors = stereoCamera.PointCloudGenerator(img_left, img_right)
        points = NormPointCloud(points)
        points, colors = DepthFilter(
            points, colors, threshold_low=-500, threshold_high=500
        )
        stereoCamera.PointCloudVisualization(points, colors)
        break
