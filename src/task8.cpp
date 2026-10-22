#include "common.hpp"
#include <opencv2/aruco.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdio>

struct BoardConfig {
    float tagSize = 0.12f;
    int expectedCount = 4;
    std::vector<int> expectedIds;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size calibSize;
};

struct BoardResult {
    bool complete = false;
    int detected = 0;
    int expected = 0;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat R;
    double angleDeg = 0.0;
    double tiltDeg = 0.0;
    cv::Point2f targetPixel;
    cv::Point3f targetCamera;
    std::vector<int> ids;
};

void detectTags(const cv::Mat& img, std::vector<std::vector<cv::Point2f>>& corners, std::vector<int>& ids);
std::vector<cv::Point3f> makeTagObjectPoints(float tagSize);
bool estimateTagPose(const std::vector<cv::Point2f>& corners, float tagSize, const cv::Mat& K, const cv::Mat& dist, cv::Vec3d& rvec, cv::Vec3d& tvec);
bool tagPoseConsistent(const std::vector<cv::Point2f>& corners, float tagSize, const cv::Mat& K, const cv::Mat& dist);
bool buildBoardResult(const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids, const BoardConfig& cfg, const cv::Mat& K, const cv::Mat& dist, cv::Size imageSize, BoardResult& result);
cv::Point3f pixelToCamera(const cv::Point2f& pixel, const cv::Mat& K, const cv::Mat& dist, const cv::Vec3d& rvec, const cv::Vec3d& tvec);
cv::Mat scaleCameraMatrix(const cv::Mat& K, cv::Size calibSize, cv::Size imageSize);
void drawResult(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids, const BoardResult& result, const cv::Mat& K, const cv::Mat& dist, const BoardConfig& cfg);
void printResult(const BoardResult& result);
bool loadCalib(const std::string& path, cv::Mat& K, cv::Mat& dist, cv::Size& size);

void processImage(const std::string& path, const BoardConfig& cfg) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    cv::Mat K = scaleCameraMatrix(cfg.cameraMatrix, cfg.calibSize, img.size());
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detectTags(img, corners, ids);
    BoardResult result;
    bool ok = buildBoardResult(corners, ids, cfg, K, cfg.distCoeffs, img.size(), result);
    //
    double costMs = common::tickMs(t0);
    cv::Mat copy = img.clone();
    if (ok) drawResult(copy, corners, ids, result, K, cfg.distCoeffs, cfg);
    std::cout << "cost = " << costMs << "ms" << std::endl;
    if (ok) printResult(result);
    else std::cout << "tags = " << ids.size() << "; complete = 0" << std::endl;
    auto out = common::outputPath("task8_result.png");
    cv::imwrite(out.string(), copy);
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName, const BoardConfig& cfg) {
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        auto outPath = common::outputPath(outName);
        writer.open(outPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
        if (writer.isOpened()) std::cout << "saving to: " << outPath.string() << std::endl;
    }
    cv::Mat frame;
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        double t0 = common::tick();
        cv::Mat K = scaleCameraMatrix(cfg.cameraMatrix, cfg.calibSize, frame.size());
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        detectTags(frame, corners, ids);
        BoardResult result;
        bool ok = buildBoardResult(corners, ids, cfg, K, cfg.distCoeffs, frame.size(), result);
        double costMs = common::tickMs(t0);
        if (ok) drawResult(frame, corners, ids, result, K, cfg.distCoeffs, cfg);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms; tags = " << ids.size() << "; complete = " << (ok && result.complete) << "; angle = " << (ok ? result.angleDeg : 0.0) << "deg" << std::endl;
        frameID++;
        if (writer.isOpened()) writer.write(frame);
        cv::imshow("result", frame);
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path> [tagSizeMM] [expectedCount] [expectedIds]" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    BoardConfig cfg;
    if (argc >= 3) cfg.tagSize = static_cast<float>(std::stod(argv[2]) / 1000.0);
    if (argc >= 4) cfg.expectedCount = std::stoi(argv[3]);
    if (argc >= 5) {
        std::stringstream ss(argv[4]);
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) cfg.expectedIds.push_back(std::stoi(item));
        if (!cfg.expectedIds.empty()) cfg.expectedCount = static_cast<int>(cfg.expectedIds.size());
    }
    auto calibPath = common::outputPath("calib.yml");
    if (!loadCalib(calibPath.string(), cfg.cameraMatrix, cfg.distCoeffs, cfg.calibSize)) {
        std::cerr << "[error] cannot load calibration: " << calibPath.string() << std::endl;
        return 1;
    }
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task8_camera.avi", cfg);
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task8_video.avi", cfg);
    } else {
        processImage(path, cfg);
    }
    return 0;
}

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
    return;
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

void drawResult(cv::Mat& img, const std::vector<std::vector<cv::Point2f>>& corners, const std::vector<int>& ids, const BoardResult& result, const cv::Mat& K, const cv::Mat& dist, const BoardConfig& cfg) {
    cv::aruco::drawDetectedMarkers(img, corners, ids);
    cv::drawFrameAxes(img, K, dist, result.rvec, result.tvec, cfg.tagSize);
    cv::circle(img, result.targetPixel, 6, cv::Scalar(0, 255, 255), -1);
    cv::Scalar color = result.complete ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(img, result.complete ? "COMPLETE" : "INCOMPLETE", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "tags %d/%d  angle %.2f deg  tilt %.2f deg", result.detected, result.expected, result.angleDeg, result.tiltDeg);
    cv::putText(img, buf, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    std::snprintf(buf, sizeof(buf), "target cam (%.3f, %.3f, %.3f) m", result.targetCamera.x, result.targetCamera.y, result.targetCamera.z);
    cv::putText(img, buf, cv::Point(20, 115), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    return;
}

void printResult(const BoardResult& result) {
    std::cout << "tags = " << result.detected << "/" << result.expected << "; complete = " << result.complete << std::endl;
    std::cout << "angle = " << result.angleDeg << "deg (CCW positive); tilt = " << result.tiltDeg << "deg" << std::endl;
    std::cout << "rvec = [" << result.rvec[0] << ", " << result.rvec[1] << ", " << result.rvec[2] << "]" << std::endl;
    std::cout << "tvec = [" << result.tvec[0] << ", " << result.tvec[1] << ", " << result.tvec[2] << "] m" << std::endl;
    std::cout << "target pixel = [" << result.targetPixel.x << ", " << result.targetPixel.y << "]" << std::endl;
    std::cout << "target camera = [" << result.targetCamera.x << ", " << result.targetCamera.y << ", " << result.targetCamera.z << "] m" << std::endl;
    std::cout << "board frame: origin = ref tag center; +X = tag right; +Y = tag up; +Z = tag normal; unit = m" << std::endl;
    return;
}

bool loadCalib(const std::string& path, cv::Mat& K, cv::Mat& dist, cv::Size& size) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    int w = 0;
    int h = 0;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    size = cv::Size(w, h);
    fs.release();
    return !K.empty();
}
