#include "color.hpp"
#include <vector>

bool getColorRange(const std::string& name, cv::Scalar& lo1, cv::Scalar& hi1, cv::Scalar& lo2, cv::Scalar& hi2, bool& hasSecond) {
    hasSecond = false;
    if (name == "red") {
        lo1 = cv::Scalar(0, 50, 50); hi1 = cv::Scalar(10, 255, 255);
        lo2 = cv::Scalar(170, 50, 50); hi2 = cv::Scalar(179, 255, 255);
        hasSecond = true;
    } else if (name == "orange") {
        lo1 = cv::Scalar(11, 50, 50); hi1 = cv::Scalar(25, 255, 255);
    } else if (name == "yellow") {
        lo1 = cv::Scalar(26, 50, 50); hi1 = cv::Scalar(34, 255, 255);
    } else if (name == "green") {
        lo1 = cv::Scalar(35, 50, 50); hi1 = cv::Scalar(85, 255, 255);
    } else if (name == "cyan") {
        lo1 = cv::Scalar(86, 50, 50); hi1 = cv::Scalar(99, 255, 255);
    } else if (name == "blue") {
        lo1 = cv::Scalar(100, 50, 50); hi1 = cv::Scalar(124, 255, 255);
    } else if (name == "purple") {
        lo1 = cv::Scalar(125, 50, 50); hi1 = cv::Scalar(155, 255, 255);
    } else if (name == "pink") {
        lo1 = cv::Scalar(156, 50, 50); hi1 = cv::Scalar(169, 255, 255);
    } else {
        return false;
    }
    return true;
}

ColorResult detectColor(const cv::Mat& bgr, const std::string& color) {
    ColorResult r;
    cv::Scalar lo1, hi1, lo2, hi2;
    bool hasSecond = false;
    if (!getColorRange(color, lo1, hi1, lo2, hi2, hasSecond)) {
        std::cerr << "[error] unknown color: " << color << std::endl;
        return r;
    }
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, lo1, hi1, mask);
    if (hasSecond) {
        cv::Mat mask2;
        cv::inRange(hsv, lo2, hi2, mask2);
        cv::bitwise_or(mask, mask2, mask);
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double maxArea = 0.0;
    int best = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a >= 300.0 && a > maxArea) {
            maxArea = a;
            best = static_cast<int>(i);
        }
    }
    if (best < 0) return r;
    cv::Moments m = cv::moments(contours[best]);
    if (m.m00 != 0) {
        r.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
    }
    r.valid = true;
    r.area = maxArea;
    return r;
}
