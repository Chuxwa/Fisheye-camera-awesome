import cv2
import glob
from calibration.opencv_methods import (
    CreateObjectPoints,
    FisheyeCalibrate,
    CameraCalibrate,
    FindImagePoints,
    SaveInternalConfig,
    UndistortImage,
    CameraUndistortImage,
)

if __name__ == "__main__":
    CHECKERBOARD = (7, 10)
    name = "right"
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    objp = CreateObjectPoints(CHECKERBOARD)
    imgpath = glob.glob("image/epipolar/" + name + "/*.png")
    objpoints, imgpoints, imgshape = FindImagePoints(
        CHECKERBOARD, imgpath, objp, subpix_criteria
    )
    fisheye = False
    if fisheye:
        K, D = FisheyeCalibrate(
            objpoints, imgpoints, subpix_criteria, calibration_flags, imgshape
        )
    else:
        K, D = CameraCalibrate(
            objpoints, imgpoints, subpix_criteria, calibration_flags, imgshape
        )

    SaveInternalConfig(K, D, name)

    imgpath = glob.glob("image/epipolar/right/*.png")
    name = "right"
    for i, frame in enumerate(imgpath):
        if fisheye:
            img, unimg = UndistortImage(frame, K, D)
        else:
            img, unimg = CameraUndistortImage(frame, K, D)

        cv2.imwrite("output/calibration/" + name + "/" + str(i) + ".png", img)
        cv2.imwrite("output/calibration/" + name + "/" + str(i) + ".png", unimg)
