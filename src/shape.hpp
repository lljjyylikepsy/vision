#pragma once
#include "common.hpp"
#include <vector>

struct ShapeResult {
    int valid = 0;
    std::string label;
    cv::Point2f center{0, 0};
    double area = 0.0;
};

std::vector<ShapeResult> detectShapes(const cv::Mat& img, double minArea);
