#include "board.hpp"
#include <opencv2/aruco.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

namespace {

void detectTags(const cv::Mat& img, std::vector<std::vector<cv::Point2f>>& corners, std::vector<int>& ids) {
    static cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    static cv::aruco::DetectorParameters params = [] {
        cv::aruco::DetectorParameters p;
        p.adaptiveThreshWinSizeMin = 3;
        p.adaptiveThreshWinSizeMax = 23;
        p.adaptiveThreshWinSizeStep = 10;
        p.minMarkerPerimeterRate = 0.01;
        p.maxMarkerPerimeterRate = 4.0;
        p.polygonalApproxAccuracyRate = 0.03;
        p.minDistanceToBorder = 3;
        p.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        p.cornerRefinementWinSize = 5;
        p.cornerRefinementMaxIterations = 50;
        p.cornerRefinementMinAccuracy = 0.01;
        return p;
    }();
    static cv::aruco::ArucoDetector detector(dict, params);
    std::vector<std::vector<cv::Point2f>> rejected;
    detector.detectMarkers(img, corners, ids, rejected);
}

std::vector<cv::Point3f> makeTagObjectPoints(float tagSize) {
    float h = tagSize / 2.0f;
    return {cv::Point3f(-h, h, 0), cv::Point3f(h, h, 0), cv::Point3f(h, -h, 0), cv::Point3f(-h, -h, 0)};
}

bool estimateTagPose(const std::vector<cv::Point2f>& corners, float tagSize, const cv::Mat& K, const cv::Mat& dist, cv::Vec3d& rvec, cv::Vec3d& tvec) {
    std::vector<cv::Point3f> obj = makeTagObjectPoints(tagSize);
    bool ok = cv::solvePnP(obj, corners, K, dist, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
    if (!ok) ok = cv::solvePnP(obj, corners, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    return ok;
}

bool tagPoseConsistent(const std::vector<cv::Point2f>& corners, float tagSize, const cv::Mat& K, const cv::Mat& dist) {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    if (!estimateTagPose(corners, tagSize, K, dist, rvec, tvec)) return false;
    std::vector<cv::Point3f> obj = makeTagObjectPoints(tagSize);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(obj, rvec, tvec, K, dist, projected);
    double err = cv::norm(corners, projected, cv::NORM_L2) / 4.0;
    return err < 1.0;
}

bool buildBoardResult(const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids, const BoardConfig& cfg, const cv::Mat& K, const cv::Mat& dist, cv::Size imageSize, BoardResult& result) {
    result.detected = static_cast<int>(ids.size());
    result.expected = cfg.expectedCount;
    result.ids = ids;
    if (ids.empty()) return false;
    int ref = 0;
    for (size_t i = 1; i < ids.size(); ++i) if (ids[i] < ids[ref]) ref = static_cast<int>(i);
    if (!estimateTagPose(corners[ref], cfg.tagSize, K, dist, result.rvec, result.tvec)) return false;
    cv::Rodrigues(result.rvec, result.R);
    result.angleDeg = -std::atan2(result.R.at<double>(1, 0), result.R.at<double>(0, 0)) * 180.0 / CV_PI;
    double nz = std::fabs(result.R.at<double>(2, 2));
    if (nz > 1.0) nz = 1.0;
    result.tiltDeg = std::acos(nz) * 180.0 / CV_PI;
    result.complete = static_cast<int>(ids.size()) >= cfg.expectedCount;
    for (int id : cfg.expectedIds) if (std::find(ids.begin(), ids.end(), id) == ids.end()) { result.complete = false; break; }
    for (const auto& tag : corners) for (const auto& p : tag) if (p.x <= 0 || p.y <= 0 || p.x >= imageSize.width - 1 || p.y >= imageSize.height - 1) result.complete = false;
    for (const auto& tag : corners) if (!tagPoseConsistent(tag, cfg.tagSize, K, dist)) result.complete = false;
    cv::Point2f sum(0, 0);
    int count = 0;
    for (const auto& tag : corners) for (const auto& p : tag) { sum += p; ++count; }
    result.targetPixel = count > 0 ? cv::Point2f(sum.x / count, sum.y / count) : cv::Point2f(imageSize.width / 2.0f, imageSize.height / 2.0f);
    result.targetCamera = pixelToCamera(result.targetPixel, K, dist, result.rvec, result.tvec);
    return true;
}

}

cv::Mat scaleCameraMatrix(const cv::Mat& K, cv::Size calibSize, cv::Size imageSize) {
    if (K.empty() || calibSize.empty() || imageSize.empty() || calibSize == imageSize) return K;
    cv::Mat out = K.clone();
    double sx = static_cast<double>(imageSize.width) / calibSize.width;
    double sy = static_cast<double>(imageSize.height) / calibSize.height;
    out.at<double>(0, 0) *= sx;
    out.at<double>(1, 1) *= sy;
    out.at<double>(0, 2) *= sx;
    out.at<double>(1, 2) *= sy;
    return out;
}

cv::Point3f pixelToCamera(const cv::Point2f& pixel, const cv::Mat& K, const cv::Mat& dist, const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    std::vector<cv::Point2f> src{pixel};
    std::vector<cv::Point2f> und;
    cv::undistortPoints(src, und, K, dist);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Vec3d n(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
    cv::Vec3d t(tvec[0], tvec[1], tvec[2]);
    cv::Vec3d ray(und[0].x, und[0].y, 1.0);
    double denom = n.dot(ray);
    if (std::fabs(denom) < 1e-9) return cv::Point3f(0, 0, 0);
    double scale = n.dot(t) / denom;
    cv::Vec3d p = ray * scale;
    return cv::Point3f(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]));
}

bool detectBoard(const cv::Mat& img, const BoardConfig& cfg, BoardResult& result) {
    cv::Mat K = scaleCameraMatrix(cfg.cameraMatrix, cfg.calibSize, img.size());
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detectTags(img, corners, ids);
    return buildBoardResult(corners, ids, cfg, K, cfg.distCoeffs, img.size(), result);
}
