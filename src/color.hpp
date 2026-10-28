#pragma once
#include "common.hpp"
#include <string>
#include <vector>

struct ColorResult {
    bool valid = false;
    std::string color;
    cv::Point2f center{0, 0};
    double area = 0.0;
};

bool getColorRange(const std::string& name, cv::Scalar& lo1, cv::Scalar& hi1, cv::Scalar& lo2, cv::Scalar& hi2, bool& hasSecond);
std::vector<std::string> allColors();
std::vector<ColorResult> detectColors(const cv::Mat& bgr, const std::vector<std::string>& colors, double minArea);
ColorResult detectColor(const cv::Mat& bgr, const std::string& color);
