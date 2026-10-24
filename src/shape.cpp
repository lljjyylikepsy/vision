#include "shape.hpp"

namespace {

std::string classifyShape(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double peri = cv::arcLength(contour, true);
    float radius = 0;
    cv::Point2f mc;
    cv::minEnclosingCircle(contour, mc, radius);
    double fill = (radius > 0) ? area / (CV_PI * radius * radius) : 0.0;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);
    int n = static_cast<int>(approx.size());
    if (fill > 0.85) return "circle";
    if (n == 3) return "triangle";
    if (n == 4) return "rectangle";
    return "unknown";
}

}

std::vector<ShapeResult> detectShapes(const cv::Mat& img, double minArea) {
    std::vector<ShapeResult> results;
    cv::Mat grey;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(grey, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    int bw = 0;
    int bt = 0;
    for (int x = 0; x < binary.cols; ++x) {
        bw += binary.at<uchar>(0, x) > 0;
        bw += binary.at<uchar>(binary.rows - 1, x) > 0;
        bt += 2;
    }
    for (int y = 0; y < binary.rows; ++y) {
        bw += binary.at<uchar>(y, 0) > 0;
        bw += binary.at<uchar>(y, binary.cols - 1) > 0;
        bt += 2;
    }
    if (bw * 2 > bt) cv::bitwise_not(binary, binary);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a < minArea) continue;
        ShapeResult r;
        r.label = classifyShape(contours[i]);
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 != 0) {
            r.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
        }
        r.valid = 1;
        r.area = a;
        results.push_back(r);
    }
    return results;
}
