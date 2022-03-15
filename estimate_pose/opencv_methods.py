import cv2
import yaml
import numpy as np
from calibration.opencv_methods import ReadInternalConfig


def ChooseGoodMatch(matches, source_sift_kp, target_sift_kp):
    goodmatches = []
    source_pts = []
    target_pts = []
    # 根据Lowe的论文进行比率测试
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            goodmatches.append(m)
            source_pts.append(source_sift_kp[m.queryIdx].pt)
            target_pts.append(target_sift_kp[m.trainIdx].pt)

    source_pts = np.int32(source_pts)
    target_pts = np.int32(target_pts)
    F, mask = cv2.findFundamentalMat(source_pts, target_pts, cv2.FM_LMEDS)
    source_pts = source_pts[mask.ravel() == 1]
    target_pts = target_pts[mask.ravel() == 1]

    return goodmatches, source_pts, target_pts


def SiftMatch(source_img, target_img):
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    source_sift_kp, source_sift_des = sift.detectAndCompute(source_img, None)
    target_sift_kp, target_sift_des = sift.detectAndCompute(target_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(source_sift_des, target_sift_des, k=2)
    return matches, source_sift_kp, target_sift_kp


def DrawEpiline(goodmatches, source_img, source_pts, target_img, target_pts):
    goodmatches = np.expand_dims(goodmatches, 1)
    idx = np.random.randint(0, len(goodmatches), size=20)
    img_out = cv2.drawMatchesKnn(
        source_img,
        source_pts,
        target_img,
        target_pts,
        goodmatches[idx],
        None,
        flags=2,
    )
    cv2.imwrite("output/sift/imgmatch.png", img_out)


def estimate_pose(source_pts, target_pts, K_source, K_target, thresh, conf=0.99999):
    if len(source_pts) < 5:
        return None
    # normalize keypoints
    source_pts = (source_pts - K_source[[0, 1], [2, 2]][None]) / K_source[
        [0, 1], [0, 1]
    ][None]
    target_pts = (target_pts - K_target[[0, 1], [2, 2]][None]) / K_target[
        [0, 1], [0, 1]
    ][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean(
        [K_source[0, 0], K_target[1, 1], K_source[0, 0], K_target[1, 1]]
    )

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        source_pts,
        target_pts,
        np.eye(3),
        threshold=ransac_thr,
        prob=conf,
        method=cv2.RANSAC,
    )
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, source_pts, target_pts, np.eye(3), 1e9, mask=mask
        )
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def SaveExteriorConfig(ret, source_name, target_name):

    response = {}
    response["rotation_matrix"] = {"data": ret[0].tolist()}
    response["translation_matrix"] = {"data": ret[1].tolist()}
    with open(
        "output/exterior_config/" + source_name + "-->" + target_name + "_config.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.dump(data=response, stream=f, allow_unicode=True)


def ReadExteriorConfig(source_name, target_name):

    with open(
        "output/exterior_config/" + source_name + "-->" + target_name + "_config.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        context = yaml.load(f, Loader=yaml.FullLoader)
    R = np.array(context["rotation_matrix"]["data"])
    T = np.array(context["translation_matrix"]["data"])
    return R, T


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
    ret = estimate_pose(
        source_pts, target_pts, K_source, K_target, thresh=1, conf=0.99999
    )
    SaveExteriorConfig(ret, source_name, target_name)
