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
    double pinkArea = 0.0;
    double blueArea = 0.0;
    cv::Mat region;
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
    if (!r.region.empty()) {
        auto regionOut = common::outputPath("task9_arrow_region.png");
        cv::imwrite(regionOut.string(), r.region);
        std::cout << "saving to: " << regionOut.string() << std::endl;
        cv::imshow("arrow region", r.region);
    }
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
        if (!cap.read(frame) || frame.empty()) break;
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
        if (!r.region.empty()) cv::imshow("arrow region", r.region);
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

void drawResult(cv::Mat& img, const ArrowResult& r) {
    if (!r.valid) {
        cv::putText(img, "ARROW NOT FOUND", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        return;
    }
    cv::circle(img, r.center, 8, cv::Scalar(0, 255, 255), 2);
    cv::circle(img, r.tip, 8, cv::Scalar(0, 0, 255), 2);
    cv::arrowedLine(img, r.center, r.tip, cv::Scalar(0, 255, 0), 3);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "angle %.2f deg (up=0, CCW positive)  length %.1f px", r.angleDeg, r.length);
    cv::putText(img, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    std::snprintf(buf, sizeof(buf), "center (%.1f, %.1f)  tip (%.1f, %.1f)", r.center.x, r.center.y, r.tip.x, r.tip.y);
    cv::putText(img, buf, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    if (!r.region.empty()) {
        std::vector<std::vector<cv::Point>> regionContours;
        cv::findContours(r.region.clone(), regionContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(img, regionContours, -1, cv::Scalar(255, 0, 255), 2);
    }
    return;
}

void printResult(const ArrowResult& r, double costMs) {
    if (!r.valid) {
        std::cout << "valid = 0; cost = " << costMs << "ms" << std::endl;
        return;
    }
    std::cout << "valid = 1; center = (" << r.center.x << ", " << r.center.y << "); tip = (" << r.tip.x << ", " << r.tip.y << "); angle = " << r.angleDeg << "deg (up=0, CCW positive); length = " << r.length << "px; pinkArea = " << r.pinkArea << "; blueArea = " << r.blueArea << "; cost = " << costMs << "ms" << std::endl;
    return;
}
