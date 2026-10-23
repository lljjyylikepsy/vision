#include "common.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

struct ArrowResult {
    bool valid = false;
    cv::Point2f center;
    cv::Point2f tip;
    double angleDeg = 0.0;
    double length = 0.0;
    double redArea = 0.0;
    double blueArea = 0.0;
};

int largestContourIndex(const std::vector<std::vector<cv::Point>>& contours);
ArrowResult detectArrow(const cv::Mat& img);
void drawResult(cv::Mat& img, const ArrowResult& r);
void printResult(const ArrowResult& r, double costMs);

void processImage(const std::string& path) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    ArrowResult r = detectArrow(img);
    //
    double costMs = common::tickMs(t0);
    cv::Mat copy = img.clone();
    drawResult(copy, r);
    printResult(r, costMs);
    auto out = common::outputPath("task9_result.png");
    cv::imwrite(out.string(), copy);
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName) {
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        auto outPath = common::outputPath(outName);
        writer.open(outPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
        if (writer.isOpened()) std::cout << "saving to: " << outPath.string() << std::endl;
    }
    cv::Mat frame;
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        double t0 = common::tick();
        //
        ArrowResult r = detectArrow(frame);
        //
        double costMs = common::tickMs(t0);
        drawResult(frame, r);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms; valid = " << r.valid << "; angle = " << r.angleDeg << "deg; length = " << r.length << "px" << std::endl;
        frameID++;
        if (writer.isOpened()) writer.write(frame);
        cv::imshow("result", frame);
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task9_camera.avi");
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task9_video.avi");
    } else {
        processImage(path);
    }
    return 0;
}

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
    r.angleDeg = std::atan2(dx, -dy) * 180.0 / CV_PI;
    r.valid = true;
    return r;
}

void drawResult(cv::Mat& img, const ArrowResult& r) {
    if (!r.valid) {
        cv::putText(img, "ARROW NOT FOUND", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        return;
    }
    cv::circle(img, r.center, 8, cv::Scalar(0, 255, 255), 2);
    cv::circle(img, r.tip, 8, cv::Scalar(0, 0, 255), 2);
    cv::arrowedLine(img, r.center, r.tip, cv::Scalar(0, 255, 0), 3);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "angle %.2f deg (up=0, CW positive)  length %.1f px", r.angleDeg, r.length);
    cv::putText(img, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    std::snprintf(buf, sizeof(buf), "center (%.1f, %.1f)  tip (%.1f, %.1f)", r.center.x, r.center.y, r.tip.x, r.tip.y);
    cv::putText(img, buf, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    return;
}

void printResult(const ArrowResult& r, double costMs) {
    if (!r.valid) {
        std::cout << "valid = 0; cost = " << costMs << "ms" << std::endl;
        return;
    }
    std::cout << "valid = 1; center = (" << r.center.x << ", " << r.center.y << "); tip = (" << r.tip.x << ", " << r.tip.y << "); angle = " << r.angleDeg << "deg (up=0, CW positive); length = " << r.length << "px; redArea = " << r.redArea << "; blueArea = " << r.blueArea << "; cost = " << costMs << "ms" << std::endl;
    return;
}
