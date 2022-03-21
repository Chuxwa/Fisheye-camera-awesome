import cv2
import glob

# from epipolar.opencv_methods import FisheyeStereoCalibrate, EpipolarRecitification
from epipolar.camera import CameraStereoCalibrate, EpipolarRecitification


if __name__ == "__main__":
    CHECKERBOARD = (8, 11)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 45, 1e-8)
    calibration_flags = cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_FIX_PRINCIPAL_POINT
    stereo_calibration_flags = (
        cv2.CALIB_THIN_PRISM_MODEL
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_INTRINSIC
        | cv2.CALIB_SAME_FOCAL_LENGTH
    )
    leftpath = "image/calibration/left/*.png"
    rightpath = "image/calibration/right/*.png"

    camera_info = CameraStereoCalibrate(
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

    test_left = glob.glob("image/calibration/left/*.png")
    test_right = glob.glob("image/calibration/right/*.png")

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
