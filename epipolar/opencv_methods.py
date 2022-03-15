import cv2
import numpy as np
import time
import os
import glob
import yaml
from calibration.opencv_methods import SingleCameraCalibrate


def SaveCameraConfig(imgshape, K, D, R, P, name):
    """
    save the camera information
    :param imgshape: list or tuple, the shape of the image
    :param K: 3x3 floating-point camera matrix
    :param D: vector of distortion coefficients
    :param R: Rectification transformation in the object space
    :param P: New camera matrix (3x3) or new projection matrix (3x4)
    :param name: str, the name of the camera config file

    """
    response = {}
    response["image_width"] = imgshape[1]
    response["image_height"] = imgshape[0]
    response["camera_matrix"] = {"data": K.tolist()}
    response["distortion_coefficients"] = {"data": D.tolist()}
    response["rectification_matrix"] = {"data": R.tolist()}
    response["projection_matrix"] = {"data": P.tolist()}
    response["distortion_model"] = cv2.CV_16SC2
    with open("output/camera_config/" + name + "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data=response, stream=f, allow_unicode=True)


def FisheyeStereoCalibrate(
    CKBD,
    leftpath,
    rightpath,
    subpix_criteria,
    calibration_flags,
    stereo_calibration_flags,
):
    """stereo calibration for fisheye camera

    Args:
        CKBD (tuple): the size of the checkerboard
        leftpath (list): the list of the image pathes
        rightpath (list): the list of the image pathes
        subpix_criteria (tuple): how to find the chess board corners
        calibration_flags (int): how to compute calibration matrix
        stereo_calibration_flags (int): how to compute stereo calibration matrix

    Returns:
        K_left (array): 3x3 floating-point camera matrix
        D_left (array): vector of distortion coefficients
        R1 (array): Rectification transformation in the object space
        P1 (array): New camera matrix (3x3) or new projection matrix (3x4)
        K_right (array): 3x3 floating-point camera matrix
        D_right (array): vector of distortion coefficients
        R2 (array): Rectification transformation in the object space
        P2 (array): New camera matrix (3x3) or new projection matrix (3x4)
        Q (array):
        imgshape (tuple): the shape of the image
    """
    _, _, imgpoints_left, K_left, D_left = SingleCameraCalibrate(
        CKBD, subpix_criteria, calibration_flags, leftpath
    )
    imgshape, objpoints, imgpoints_right, K_right, D_right = SingleCameraCalibrate(
        CKBD, subpix_criteria, calibration_flags, rightpath
    )
    R = np.zeros((1, 1, 3), dtype=np.float64)
    T = np.zeros((1, 1, 3), dtype=np.float64)

    _, K_left, D_left, K_right, D_right, R, T = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_left,
        D_left,
        K_right,
        D_right,
        imgshape,
        R,
        T,
        stereo_calibration_flags,
    )

    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        imgshape,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
    )
    SaveCameraConfig(imgshape, K_left, D_left, R1, P1, "left_")
    SaveCameraConfig(imgshape, K_right, D_right, R2, P2, "right_")

    return K_left, D_left, R1, P1, K_right, D_right, R2, P2, Q, imgshape


def ReadCameraConfig(name):
    """read camera configuration from yaml file

    Args:
        name (str): the name of yaml file

    Returns:
        K (array): 3x3 floating-point camera matrix
        D (array): vector of distortion coefficients
        R (array): Rectification transformation in the object space
        P (array): New camera matrix (3x3) or new projection matrix (3x4)
    """
    with open("output/camera_config" + name + "config.yaml", "r", encoding="utf-8") as f:
        context = yaml.load(f, Loader=yaml.FullLoader)
    K = np.array(context["camera_matrix"]["data"])
    D = np.array(context["distortion_coefficients"]["data"])
    P = np.array(context["rectification_matrix"]["data"])
    R = np.array(context["projection_matrix"]["data"])
    return K, D, P, R


def EpipolarRecitification(leftcamera, rightcamera, img_left, img_right):
    """epipolar recitification

    Args:
        leftcamera (tuple): the informations of the left camera
        rightcamera (tuple): the informations of the right camera
        img_left (array): the left image
        img_right (array): the right image

    Returns:
        result: the source image
        result_left: the recitification left image
        result_right: the recitification right image
    """
    K_left, D_left, R1, P1, _ = leftcamera
    K_right, D_right, R2, P2, imgshape = rightcamera
    map1_1, map1_2 = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, R1, P1, imgshape, cv2.CV_16SC2
    )
    map2_1, map2_2 = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, R2, P2, imgshape, cv2.CV_16SC2
    )
    result_left = cv2.remap(
        img_left,
        map1_1,
        map1_2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    result_right = cv2.remap(
        img_right,
        map2_1,
        map2_2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    result = np.concatenate((result_left, result_right), axis=1)
    result[::20, :] = 0
    return result, result_left, result_right


if __name__ == "__main__":
    CHECKERBOARD = (7, 10)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    stereo_calibration_flags = (
        cv2.fisheye.CALIB_FIX_INTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    leftpath = "image/calibration/left/*.png"
    rightpath = "image/calibration/right/*.png"

    (
        K_left,
        D_left,
        R1,
        P1,
        K_right,
        D_right,
        R2,
        P2,
        Q,
        imgshape,
    ) = FisheyeStereoCalibrate(
        CHECKERBOARD,
        leftpath,
        rightpath,
        subpix_criteria,
        calibration_flags,
        stereo_calibration_flags,
    )
    leftcamera = (K_left, D_left, R1, P1, imgshape)
    rightcamera = (K_right, D_right, R2, P2, imgshape)

    test_left = glob.glob("image/test/left/*.png")
    test_right = glob.glob("image/test/right/*.png")

    for i in range(len(test_left)):
        img_left = cv2.imread(test_left[i])
        img_right = cv2.imread(test_right[i])

        result, result_left, result_right = EpipolarRecitification(
            leftcamera,
            rightcamera,
            img_left,
            img_right,
        )
        cv2.imwrite("output/epipolar/rec" + str(i) + ".png", result)
        cv2.imwrite("output/epipolar/rec_left" + str(i) + ".png", result_left)
        cv2.imwrite("output/epipolar/rec_right" + str(i) + ".png", result_right)
