#pragma once
#include "common.hpp"

struct ArrowResult {
    bool valid = false;
    cv::Point2f center;
    cv::Point2f tip;
    double angleDeg = 0.0;
    double length = 0.0;
    double pinkArea = 0.0;
    double blueArea = 0.0;
    cv::Mat region;
};

ArrowResult detectArrow(const cv::Mat& img);
