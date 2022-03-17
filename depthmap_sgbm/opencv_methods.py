import cv2
import numpy as np
import glob
from epipolar.opencv_methods import (
    ReadCameraConfig,
    EpipolarRecitification,
    ReadReprejectionMatrix,
)


class StereoDepthEstimation(object):
    """Stereo camera for depth estimation

    Args:
        imgshape (tuple): the shape of the image
        camera1_name (str): the name of the camera one
        camera2_name (str): the name of the camera two
        blockSize (int, optional): the size of matching block. Defaults to 3.
        disparity (int, optional): the number of matching disparity. Defaults to 16.
    """

    def __init__(
        self,
        imgshape: tuple,
        camera1_name: str,
        camera2_name: str,
        blockSize: int = 3,
        disparity: int = 16,
    ):
        """Init function of stereo camera for depth estimation

        Args:
            imgshape (tuple): the shape of the image
            camera1_name (str): the name of the camera one
            camera2_name (str): the name of the camera two
            blockSize (int, optional): the size of matching block. Defaults to 3.
            disparity (int, optional): the number of matching disparity. Defaults to 16.
        """
        self.imgshape = imgshape
        self.camera1_name = camera1_name
        self.camera2_name = camera2_name
        K1, D1, R1, P1 = ReadCameraConfig(self.camera1_name)
        K2, D2, R2, P2 = ReadCameraConfig(self.camera2_name)
        self.Q = ReadReprejectionMatrix(self.camera1_name + "-" + self.camera2_name)
        self.camera1_cfg = (K1, D1, R1, P1, self.imgshape)
        self.camera2_cfg = (K2, D2, R2, P2, self.imgshape)
        self.blockSize = blockSize
        self.disparity = disparity

        self.depthmap_func = cv2.StereoSGBM_create(
            minDisparity=-1 * self.disparity,
            numDisparities=5 * self.disparity,
            blockSize=self.blockSize,
            P1=8 * 4 * self.blockSize ** 2,
            P2=32 * 4 * self.blockSize ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            # preFilterCap=63,
            # speckleWindowSize=200,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def ComputerDepthMap(self, imgleft, imgright):
        """Use Opencv finction to compute depth map

        Args:
            imgleft (ndarray): the left image
            imgright (ndarray): the right image

        Returns:
            unnorm_disp (ndarray): the unnormlized depth map
            norm_disp (ndarray): the normlized depth map
        """
        unnorm_disp = self.depthmap_func.compute(imgleft, imgright)

        norm_disp = cv2.normalize(
            unnorm_disp,
            unnorm_disp,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        return unnorm_disp, norm_disp

    def DepthEstimation(self, unnorm_disp, scale: float = 1.0):
        """Reproject Image To 3D points

        Args:
            unnorm_disp (ndarray): the unnormlized disparity map
            scale (float): the scale factor of the depth
        Returns:
            points_3d (ndarray): the positions of the points
            depth (ndarray): the depth of all points
        """
        points_3d = cv2.reprojectImageTo3D(unnorm_disp, self.Q)
        _, _, depth = cv2.split(points_3d)
        depth = depth * scale
        depth = np.asarray(depth, dtype=np.float32)
        return points_3d, depth


if __name__ == "__main__":
    imgshape = (1280, 720)
    camera1_name = "left"
    camera2_name = "right"
    stereoCamera = StereoDepthEstimation(imgshape, camera1_name, camera2_name)

    test_left = glob.glob("image/test/left/*.png")
    test_right = glob.glob("image/test/right/*.png")

    for i in range(len(test_left)):
        img_left = cv2.imread(test_left[i])
        img_right = cv2.imread(test_right[i])

        result, result_left, result_right = EpipolarRecitification(
            stereoCamera.camera1_cfg,
            stereoCamera.camera2_cfg,
            img_left,
            img_right,
        )
        imgL = cv2.cvtColor(result_left, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(result_right, cv2.COLOR_BGR2GRAY)
        disp = StereoDepthEstimation.ComputerDepthMap(imgL, imgR)

        cv2.imwrite("output/sgbm/depthmap_" + str(i) + ".png", disp)

