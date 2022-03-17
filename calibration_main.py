import cv2
import glob
from calibration.opencv_methods import (
    CreateObjectPoints,
    FisheyeCalibrate,
    FindImagePoints,
    SaveInternalConfig,
    UndistortImage,
)

if __name__ == "__main__":
    CHECKERBOARD = (24, 24)
    name = "test"
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.1)
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    objp = CreateObjectPoints(CHECKERBOARD)
    imgpath = glob.glob("image/calibration/" + name + "/*.png")
    objpoints, imgpoints, imgshape = FindImagePoints(
        CHECKERBOARD, imgpath, objp, subpix_criteria
    )
    K, D = FisheyeCalibrate(objpoints, imgpoints, calibration_flags, imgshape)
    SaveInternalConfig(K, D, name)

    imgpath = glob.glob("image/source/around_views/*.png")
    name = "around_views"
    for i, frame in enumerate(imgpath):
        img, unimg = UndistortImage(frame, K, D)
        cv2.imwrite("output/calibration/" + name + "/" + str(i) + ".png", img)
        cv2.imwrite("output/calibration/" + name + "/" + str(i) + ".png", unimg)
