#pragma once
#include "common.hpp"

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

bool detectBoard(const cv::Mat& img, const BoardConfig& cfg, BoardResult& result);
cv::Point3f pixelToCamera(const cv::Point2f& pixel, const cv::Mat& K, const cv::Mat& dist, const cv::Vec3d& rvec, const cv::Vec3d& tvec);
cv::Mat scaleCameraMatrix(const cv::Mat& K, cv::Size calibSize, cv::Size imageSize);
