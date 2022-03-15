import cv2
from calibration.opencv_methods import ReadInternalConfig
from estimate_pose.opencv_methods import (
    SiftMatch,
    ChooseGoodMatch,
    DrawEpiline,
    EstimatePose,
    SaveExteriorConfig,
)

if __name__ == "__main__":
    source_name = "front_right"
    target_name = "front_left"
    source_path = "output/calibration/" + source_name + "/41.png"
    target_path = "output/calibration/" + target_name + "/41.png"
    source_img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    matches, source_kps, target_kps = SiftMatch(source_img, target_img)
    goodmatches, source_pts, target_pts = ChooseGoodMatch(
        matches, source_kps, target_kps
    )
    DrawEpiline(goodmatches, source_img, source_kps, target_img, target_kps)
    K_source, D_source = ReadInternalConfig(source_name)
    K_target, D_target = ReadInternalConfig(target_name)
    ret = EstimatePose(
        source_pts, target_pts, K_source, K_target, thresh=1, conf=0.99999
    )
    SaveExteriorConfig(ret, source_name, target_name)
