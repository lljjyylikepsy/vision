#pragma once
#include "common.hpp"
#include <string>

struct TargetSpec {
    std::string color;
    std::string shape;
    std::string digit;
};

struct TargetResult {
    bool valid = false;
    std::string label;
    cv::Point2f center{0, 0};
    double confidence = 0.0;
};

bool loadTargetSpec(const std::string& path, TargetSpec& spec);
TargetResult detectTarget(const cv::Mat& img, const TargetSpec& spec);
