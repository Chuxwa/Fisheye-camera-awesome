import cv2
import glob
import yaml
import numpy as np


def CreateObjectPoints(CKBD: tuple):
    """
    :param CKBD: list or tuple, the size of the checkerboard
    reture:
        objp: the object points, 3d point in real world space
    """
    objp = np.zeros((1, CKBD[0] * CKBD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CKBD[0], 0 : CKBD[1]].T.reshape(-1, 2)
    return objp


def FindImagePoints(CKBD: tuple, imgpath: list, objp: np.array, subpix_criteria: tuple):
    """
    :param CKBD: list or tuple, the size of the checkerboard
    :param imgpath: string list, the root path of the all useful images
    :param objp: np.array, 3d point in real world space
    :param subpix_criteria: flags, how to find the chess board corners
    reture:
        objpoints: the object points, 3d point in real world space
        imgpoints: the image points, 2d points in image plane
        imgshape: list or tuple, the shape of the image
    """
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    imgshape = None
    for fname in imgpath:
        img = cv2.imread(fname)
        if imgshape == None:
            imgshape = img.shape[:2]
        else:
            assert imgshape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            CKBD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)

    return objpoints, imgpoints, imgshape[::-1]


def FisheyeCalibrate(
    objpoints: list, imgpoints: list, calibration_flags: tuple, imgshape: tuple
):
    """
    :param objpoints: the object points, 3d point in real world space
    :param imgpoints: the image points, 2d points in image plane
    :param calibration_flags: flags, how to compute calibration matrix
    :param imgshape: list or tuple, the size of the image
    return:
        K: 3x3 floating-point camera matrix
        D: vector of distortion coefficients
    """
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    _, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        imgshape,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return K, D


def UndistortImage(imgpath, K, D):
    """
    :param imgpath: string, the root path of the image
    :param K: 3x3 floating-point camera matrix
    :param D: vector of distortion coefficients
    return:
        img: the source image
        undistorted_img: the undistorted image
    """
    img = cv2.imread(imgpath)
    imgshape = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, imgshape[::-1], cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return img, undistorted_img


def SingleCameraCalibrate(CKBD, subpix_criteria, calibration_flags, rootpath):
    """
    :param CKBD: list or tuple, the size of the checkerboard
    :param subpix_criteria: flags, how to find the chess board corners
    :param calibration_flags: flags, how to compute calibration matrix
    :param rootpath: the root path of the image
    return:
        imgshape: list or tuple, the shape of the image
        objpoints: the object points, 3d point in real world space
        imgpoints: the image points, 2d points in image plane
        K: 3x3 floating-point camera matrix
        D: vector of distortion coefficients
    """
    objp = CreateObjectPoints(CKBD)
    imgpath = glob.glob(rootpath)
    objpoints, imgpoints, imgshape = FindImagePoints(
        CKBD, imgpath, objp, subpix_criteria
    )
    K, D = FisheyeCalibrate(objpoints, imgpoints, calibration_flags, imgshape)
    objpoints = np.reshape(objpoints, (-1, 1, CKBD[0] * CKBD[1], 3))
    imgpoints = np.reshape(imgpoints, (-1, 1, CKBD[0] * CKBD[1], 2))
    return imgshape, objpoints, imgpoints, K, D


def SaveInternalConfig(K, D, name):
    """
    save the camera information
    :param K: 3x3 floating-point camera matrix
    :param D: vector of distortion coefficients
    :param name: str, the name of the camera config file

    """
    response = {}
    response["camera_matrix"] = {"data": K.tolist()}
    response["distortion_coefficients"] = {"data": D.tolist()}
    with open(
        "output/internal_config/" + name + "_config.yaml", "w", encoding="utf-8"
    ) as f:
        yaml.dump(data=response, stream=f, allow_unicode=True)


def ReadInternalConfig(name):
    """read camera configuration from yaml file

    Args:
        name (str): the name of yaml file

    Returns:
        K (array): 3x3 floating-point camera matrix
        D (array): vector of distortion coefficients
    """
    with open(
        "output/internal_config/" + name + "_config.yaml", "r", encoding="utf-8"
    ) as f:
        context = yaml.load(f, Loader=yaml.FullLoader)
    K = np.array(context["camera_matrix"]["data"])
    D = np.array(context["distortion_coefficients"]["data"])
    return K, D


if __name__ == "__main__":
    CHECKERBOARD = (7, 10)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    objp = CreateObjectPoints(CHECKERBOARD)
    imgpath = glob.glob("image/calibration/left/*.png")
    objpoints, imgpoints, imgshape = FindImagePoints(
        CHECKERBOARD, imgpath, objp, subpix_criteria
    )
    K, D = FisheyeCalibrate(objpoints, imgpoints, calibration_flags, imgshape)

    imgpath = glob.glob("image/around_views/*.png")
    for i, frame in enumerate(imgpath):
        img, undistorted_img = UndistortImage(frame, K, D)
        cv2.imwrite("output/calibration/distorted" + str(i) + ".png", img)
        cv2.imwrite("output/calibration/undistorted" + str(i) + ".png", undistorted_img)
