import cv2
import numpy as np
import glob
from epipolar.opencv_methods import ReadCameraConfig, EpipolarRecitification

def ComputerDepthMap(imgleft, imgright):
    """use opencv finction to compute depth map

    Args:
        imgleft (array): the left image
        imgright (array): the right image

    Returns:
        disp (array): the depth map
    """
    blockSize = 3
    DepthMap_func = cv2.StereoSGBM_create(
        minDisparity=-16,
        numDisparities=5 * 16,
        blockSize=blockSize,
        P1=8 * 4 * blockSize ** 2,
        P2=32 * 4 * blockSize ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        # preFilterCap=63,
        # speckleWindowSize=200,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = DepthMap_func.compute(imgleft, imgright)

    disp_img = cv2.normalize(
        disparity,
        disparity,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    return disparity,disp_img


def DepthEstimation(disparity, Q, scale:float=1.0, method:bool=True):
    """reproject Image To 3D points

    Args:
        disparity (array): the disparity map
        Q (array): disparity-to-depth mapping matrix (4x4)
        scale: float32 
    Returns:
        depth (array): the depth of all points
    """
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    if method:
        points_3d = cv2.reprojectImageTo3D(disparity, Q)  # 单位是毫米(mm)
        x, y, depth = cv2.split(points_3d)
    else:
        # baseline = abs(camera_config["T"][0])
        baseline = 1 / Q[3, 2]  # 基线也可以由T[0]计算
        fx = abs(Q[2, 3])
        depth = (fx * baseline) / disparity
    depth = depth * scale
    # depth = np.asarray(depth, dtype=np.uint16)
    depth = np.asarray(depth, dtype=np.float32)
    return points_3d, depth

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
