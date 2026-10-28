#include "common.hpp"
#include <vector>

struct DetectResult {
    bool valid = 0;
    std::string color;
    cv::Point2f center = {0, 0};
    double area = 0.0;
};

bool getColorRange(const std::string& name, cv::Scalar& lo1, cv::Scalar& hi1, cv::Scalar& lo2, cv::Scalar& hi2, bool& hasSecond);
std::vector<std::string> allColors();
std::vector<DetectResult> detectColors(const cv::Mat& bgr, const std::vector<std::string>& colors, double minArea);
void drawResults(cv::Mat& img, const std::vector<DetectResult>& results);

void processImage(const std::string& path, const std::vector<std::string>& colors) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    std::vector<DetectResult> results = detectColors(img, colors, 300.0);
    //
    double costMs = common::tickMs(t0);
    std::cout << "count = " << results.size() << "; cost = " << costMs << "ms" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "[" << i << "] color=" << results[i].color << "; center=(" << results[i].center.x << "," << results[i].center.y << "); area=" << results[i].area << std::endl;
    }
    cv::Mat copy = img.clone();
    drawResults(copy, results);
    cv::imwrite(common::outputPath("task4_result.png").string(), copy);
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName, const std::vector<std::string>& colors) {
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        writer.open(common::outputPath(outName).string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
    }
    cv::Mat frame;
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        std::vector<DetectResult> results = detectColors(frame, colors, 300.0);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; count = " << results.size();
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "; " << results[i].color << "(" << results[i].center.x << "," << results[i].center.y << ")";
        }
        std::cout << std::endl;
        frameID++;
        drawResults(frame, results);
        if (writer.isOpened()) writer.write(frame);
        cv::imshow("result", frame);
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path> [color ... | all]" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    std::vector<std::string> colors;
    if (argc < 3 || std::string(argv[2]) == "all") {
        colors = allColors();
    } else {
        for (int i = 2; i < argc; ++i) colors.push_back(argv[i]);
    }
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task4_camera.avi", colors);
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task4_video.avi", colors);
    } else {
        processImage(path, colors);
    }
    return 0;
}

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
    } else if (name == "black") {
        lo1 = cv::Scalar(0, 0, 0); hi1 = cv::Scalar(180, 255, 110);
    } else {
        return false;
    }
    return true;
}

std::vector<std::string> allColors() {
    return {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "black"};
}

std::vector<DetectResult> detectColors(const cv::Mat& bgr, const std::vector<std::string>& colors, double minArea) {
    std::vector<DetectResult> results;
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    for (const auto& color : colors) {
        cv::Scalar lo1, hi1, lo2, hi2;
        bool hasSecond = false;
        if (!getColorRange(color, lo1, hi1, lo2, hi2, hasSecond)) {
            std::cerr << "[error] unknown color: " << color << std::endl;
            continue;
        }
        cv::Mat mask;
        cv::inRange(hsv, lo1, hi1, mask);
        if (hasSecond) {
            cv::Mat mask2;
            cv::inRange(hsv, lo2, hi2, mask2);
            cv::bitwise_or(mask, mask2, mask);
        }
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (size_t i = 0; i < contours.size(); ++i) {
            double a = cv::contourArea(contours[i]);
            if (a < minArea) continue;
            DetectResult r;
            r.color = color;
            cv::Moments m = cv::moments(contours[i]);
            if (m.m00 != 0) {
                r.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
            }
            r.valid = 1;
            r.area = a;
            results.push_back(r);
        }
    }
    return results;
}

void drawResults(cv::Mat& img, const std::vector<DetectResult>& results) {
    for (size_t i = 0; i < results.size(); ++i) {
        cv::Point p(static_cast<int>(results[i].center.x), static_cast<int>(results[i].center.y));
        cv::circle(img, p, 6, cv::Scalar(0, 255, 255), 2);
        cv::putText(img, results[i].color, p + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 4);
        cv::putText(img, results[i].color, p + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
    std::string text = "count = " + std::to_string(results.size());
    cv::putText(img, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 4);
    cv::putText(img, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}
