import cv2
import glob
from epipolar.opencv_methods import FisheyeStereoCalibrate, EpipolarRecitification


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
    leftpath = "image/epipolar/left/*.png"
    rightpath = "image/epipolar/right/*.png"

    camera_info = FisheyeStereoCalibrate(
        CHECKERBOARD,
        leftpath,
        rightpath,
        subpix_criteria,
        calibration_flags,
        stereo_calibration_flags,
    )
    K_left, D_left, R1, P1, K_right, D_right, R2, P2, Q, imgshape = camera_info
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
