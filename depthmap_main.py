import cv2
import glob
from epipolar.opencv_methods import ReadCameraConfig
from sgbm.opencv_methods import EpipolarRecitification, ComputerDepthMap


if __name__ == "__main__":
    imgshape = (1280, 720)
    K_left, D_left, R1, P1 = ReadCameraConfig("left_")
    K_right, D_right, R2, P2 = ReadCameraConfig("right_")
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
        # 将图片置为灰度图，为StereoBM作准备
        imgL = cv2.cvtColor(result_left, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(result_right, cv2.COLOR_BGR2GRAY)
        disp = ComputerDepthMap(imgL, imgR)

        cv2.imwrite("output/sgbm/depthmap_" + str(i) + ".png", disp)

    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
