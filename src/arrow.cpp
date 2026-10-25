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
    cv::Mat red1;
    cv::Mat red2;
    cv::Mat red;
    cv::Mat blue;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 120, 100), cv::Scalar(25, 255, 255), red1);
    cv::inRange(hsv, cv::Scalar(170, 120, 100), cv::Scalar(180, 255, 255), red2);
    cv::bitwise_or(red1, red2, red);
    cv::inRange(hsv, cv::Scalar(90, 100, 80), cv::Scalar(140, 255, 255), blue);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(red, red, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(blue, blue, cv::MORPH_CLOSE, kernel);
    std::vector<std::vector<cv::Point>> redContours;
    std::vector<std::vector<cv::Point>> blueContours;
    cv::findContours(red, redContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::findContours(blue, blueContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    int ri = largestContourIndex(redContours);
    int bi = largestContourIndex(blueContours);
    if (ri < 0 || bi < 0) return r;
    r.redArea = cv::contourArea(redContours[ri]);
    r.blueArea = cv::contourArea(blueContours[bi]);
    double minRed = std::max(50.0, img.total() * 1e-5);
    double minBlue = std::max(100.0, img.total() * 2e-5);
    if (r.redArea < minRed || r.blueArea < minBlue) return r;
    cv::Moments mr = cv::moments(redContours[ri]);
    cv::Moments mb = cv::moments(blueContours[bi]);
    if (mr.m00 <= 0 || mb.m00 <= 0) return r;
    cv::Point2f redC(static_cast<float>(mr.m10 / mr.m00), static_cast<float>(mr.m01 / mr.m00));
    cv::Point2f blueC(static_cast<float>(mb.m10 / mb.m00), static_cast<float>(mb.m01 / mb.m00));
    cv::Point2f dir = redC - blueC;
    double dirLen = cv::norm(dir);
    if (dirLen < 1.0) return r;
    dir /= static_cast<float>(dirLen);
    double minProj = 1e18;
    double maxProj = -1e18;
    for (const auto& p : blueContours[bi]) {
        double d = (cv::Point2f(p) - blueC).dot(dir);
        minProj = std::min(minProj, d);
        maxProj = std::max(maxProj, d);
    }
    double baseCut = minProj + 0.05 * (maxProj - minProj);
    cv::Point2f sum(0, 0);
    int count = 0;
    for (const auto& p : blueContours[bi]) {
        double d = (cv::Point2f(p) - blueC).dot(dir);
        if (d <= baseCut) {
            sum += cv::Point2f(p);
            ++count;
        }
    }
    if (count == 0) return r;
    r.center = sum / static_cast<float>(count);
    cv::Mat tip1;
    cv::Mat tip2;
    cv::Mat tipMask;
    cv::inRange(hsv, cv::Scalar(0, 140, 200), cv::Scalar(25, 255, 255), tip1);
    cv::inRange(hsv, cv::Scalar(170, 140, 200), cv::Scalar(180, 255, 255), tip2);
    cv::bitwise_or(tip1, tip2, tipMask);
    tipMask &= red;
    std::vector<cv::Point> tipPoints;
    cv::findNonZero(tipMask, tipPoints);
    cv::Point2f perp(-dir.y, dir.x);
    std::vector<cv::Point2f> leftPts;
    std::vector<cv::Point2f> rightPts;
    for (const auto& p : tipPoints) {
        cv::Point2f q(p);
        if ((q - redC).dot(perp) < 0.0f) leftPts.push_back(q);
        else rightPts.push_back(q);
    }
    bool tipOk = false;
    if (leftPts.size() >= 10 && rightPts.size() >= 10) {
        cv::Vec4f l1;
        cv::Vec4f l2;
        cv::fitLine(leftPts, l1, cv::DIST_L2, 0, 0.01, 0.01);
        cv::fitLine(rightPts, l2, cv::DIST_L2, 0, 0.01, 0.01);
        cv::Point2f p1(l1[2], l1[3]);
        cv::Point2f d1(l1[0], l1[1]);
        cv::Point2f p2(l2[2], l2[3]);
        cv::Point2f d2(l2[0], l2[1]);
        double den = d1.x * d2.y - d1.y * d2.x;
        if (std::fabs(den) > 1e-6) {
            double t = ((p2.x - p1.x) * d2.y - (p2.y - p1.y) * d2.x) / den;
            cv::Point2f tip = p1 + t * d1;
            cv::Rect rb = cv::boundingRect(redContours[ri]);
            double margin = 0.2 * std::max(rb.width, rb.height);
            if (tip.x >= rb.x - margin && tip.x <= rb.x + rb.width + margin && tip.y >= rb.y - margin && tip.y <= rb.y + rb.height + margin) {
                r.tip = tip;
                tipOk = true;
            }
        }
    }
    if (!tipOk) {
        double bestProj = -1e18;
        for (const auto& p : tipPoints) {
            cv::Point2f q(p);
            double proj = (q - r.center).dot(dir);
            if (proj > bestProj) {
                bestProj = proj;
                r.tip = q;
            }
        }
        if (bestProj < -1e17) return r;
    }
    r.length = cv::norm(r.tip - r.center);
    if (r.length < 10.0) return r;
    double dx = r.tip.x - r.center.x;
    double dy = r.tip.y - r.center.y;
    r.angleDeg = std::atan2(-dx, -dy) * 180.0 / CV_PI;
    r.valid = true;
    return r;
}
