#include "arrow.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {

int largestContourIndex(const std::vector<std::vector<cv::Point>>& contours) {
    int best = -1;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            best = static_cast<int>(i);
        }
    }
    return best;
}

}

ArrowResult detectArrow(const cv::Mat& img) {
    ArrowResult r;
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat pink;
    cv::Mat blue;
    cv::Mat white;
    cv::Mat lowS;
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::inRange(hsv, cv::Scalar(148, 60, 180), cv::Scalar(175, 255, 255), pink);
    cv::inRange(hsv, cv::Scalar(100, 60, 150), cv::Scalar(145, 255, 255), blue);
    cv::threshold(channels[2], white, 245, 255, cv::THRESH_BINARY);
    cv::threshold(channels[1], lowS, 60, 255, cv::THRESH_BINARY);
    cv::bitwise_not(lowS, lowS);
    cv::bitwise_and(white, lowS, white);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(pink, pink, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(blue, blue, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(white, white, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    cv::Mat gate;
    cv::dilate(white, gate, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));
    cv::bitwise_and(blue, gate, blue);
    std::vector<std::vector<cv::Point>> pinkContours;
    std::vector<std::vector<cv::Point>> blueContours;
    cv::findContours(pink.clone(), pinkContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::findContours(blue.clone(), blueContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int pi = largestContourIndex(pinkContours);
    int bi = largestContourIndex(blueContours);
    if (pi < 0 || bi < 0) return r;
    r.pinkArea = cv::contourArea(pinkContours[pi]);
    r.blueArea = cv::contourArea(blueContours[bi]);
    double minPink = std::max(200.0, img.total() * 2e-5);
    double minBlue = std::max(300.0, img.total() * 1e-4);
    if (r.pinkArea < minPink || r.blueArea < minBlue) return r;
    cv::Moments mp = cv::moments(pinkContours[pi]);
    cv::Moments mb = cv::moments(blueContours[bi]);
    if (mp.m00 <= 0 || mb.m00 <= 0) return r;
    cv::Point2f pinkC(static_cast<float>(mp.m10 / mp.m00), static_cast<float>(mp.m01 / mp.m00));
    cv::Point2f blueC(static_cast<float>(mb.m10 / mb.m00), static_cast<float>(mb.m01 / mb.m00));
    double dist = cv::norm(pinkC - blueC);
    double diag = std::sqrt(static_cast<double>(img.cols) * img.cols + static_cast<double>(img.rows) * img.rows);
    if (dist < 20.0 || dist > 0.8 * diag) return r;
    cv::Rect overlap = cv::boundingRect(pinkContours[pi]) & cv::boundingRect(blueContours[bi]);
    if (overlap.area() <= 0 && dist > 0.15 * diag) return r;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int componentCount = cv::connectedComponentsWithStats(white, labels, stats, centroids, 8, CV_32S);
    int bestLabel = 0;
    int bestArea = 0;
    for (int i = 1; i < componentCount; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) {
            bestArea = area;
            bestLabel = i;
        }
    }
    r.region = cv::Mat::zeros(img.size(), CV_8U);
    if (bestLabel > 0) r.region = (labels == bestLabel);
    r.center = blueC;
    r.tip = pinkC;
    r.length = dist;
    double dx = r.tip.x - r.center.x;
    double dy = r.tip.y - r.center.y;
    r.angleDeg = std::atan2(-dx, -dy) * 180.0 / CV_PI;
    r.valid = true;
    return r;
}
