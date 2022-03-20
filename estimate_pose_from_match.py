import cv2
import glob
import numpy as np
from calibration.opencv_methods import ReadInternalConfig
from estimate_pose.opencv_methods import EstimatePose, SaveExteriorConfig


def ReadMatchFromTxt(filepath, goodmatches):
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")  # 去掉列表中每一个元素的换行符
            goodmatches.append(np.float_(line.split(" ")))
    return goodmatches


def UndistortImage(imgpath, K, D, P):
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
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, np.eye(3), P, imgshape[::-1], cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return img, undistorted_img


if __name__ == "__main__":
    source_name = "left"
    target_name = "right"
    K_source, D_source = ReadInternalConfig(source_name)
    K_target, D_target = ReadInternalConfig(target_name)
    pairpath = glob.glob(
        "image/calibration_pair/" + source_name + "--" + target_name + "/*.txt"
    )

    goodmatches = []
    for pair in pairpath:
        goodmatches = ReadMatchFromTxt(pair, goodmatches)
    goodmatches = np.int32(goodmatches)

    source_pts = goodmatches[:, :2]
    target_pts = goodmatches[:, 2:]
    ret = EstimatePose(
        source_pts, target_pts, K_source, K_target, thresh=1, conf=0.99999
    )
    SaveExteriorConfig(ret, source_name, target_name)
    source_M = np.hstack((ret[0], np.expand_dims(ret[1],axis = -1)))
    M = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    source_P = np.dot(K_source, source_M)
    P = np.dot(K_source, M)

    imgpath = "image/calibration/" + source_name + "/image_6.png"
    img, unimg1 = UndistortImage(imgpath, K_source, D_source, source_P)
    cv2.imwrite("output/estimate_pose/" + source_name + "/sourceimage.png", img)
    cv2.imwrite("output/estimate_pose/" + source_name + "/translationimage.png", unimg1)
    img, unimg2 = UndistortImage(imgpath, K_source, D_source, P)
    cv2.imwrite("output/estimate_pose/" + source_name + "/undistortimage.png", unimg2)

    result = np.concatenate((unimg1, unimg2), axis=1)
    result[::20, :] = 0
    cv2.imwrite("output/estimate_pose/" + source_name + "/results.png", result)

