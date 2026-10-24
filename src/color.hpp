#pragma once
#include "common.hpp"

struct ColorResult {
    bool valid = false;
    cv::Point2f center{0, 0};
    double area = 0.0;
};

bool getColorRange(const std::string& name, cv::Scalar& lo1, cv::Scalar& hi1, cv::Scalar& lo2, cv::Scalar& hi2, bool& hasSecond);
ColorResult detectColor(const cv::Mat& bgr, const std::string& color);
