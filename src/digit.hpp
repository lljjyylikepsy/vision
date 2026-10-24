#pragma once
#include "common.hpp"
#include <vector>

struct DigitResult {
    int valid = 0;
    int label = -1;
    cv::Point2f center{0, 0};
    double confidence = 0.0;
};

std::vector<DigitResult> detectDigits(const cv::Mat& img, double confMin);
