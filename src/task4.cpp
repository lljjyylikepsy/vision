#include "common.hpp"
#include <vector>

struct DetectResult {
    bool valid  = 0;
    cv::Point2f center = {0, 0};
    double area   = 0.0;
};

bool getColorRange(const std::string& name, cv::Scalar& lo1, cv::Scalar& hi1, cv::Scalar& lo2, cv::Scalar& hi2, bool& hasSecond);
DetectResult detectColor(const cv::Mat& bgr, const std::string& color);
void drawResult(cv::Mat& img, const DetectResult& r);

void processImage(const std::string& path, const std::string& color) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    DetectResult r = detectColor(img, color);
    double costMs = common::tickMs(t0);
    if (r.valid) {
        std::cout << "valid=1; center=(" << r.center.x << "," << r.center.y << "); area=" << r.area << "; cost=" << costMs << "ms" << std::endl;
    } else {
        std::cout << "valid=0; cost=" << costMs << "ms" << std::endl;
    }
    cv::Mat copy = img.clone();
    drawResult(copy, r);
    //
    cv::imwrite(common::outputPath("task4_result.png").string(), copy);
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processVideo(const std::string& path, const std::string& color) {
    auto cap = common::openVideo(path);
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        writer.open(common::outputPath("task4_video.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
    }
    cv::Mat frame;
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        DetectResult r = detectColor(frame, color);
        if (r.valid) {
            std::cout << "FrameID: " << frameID << "; valid=1; center=(" << r.center.x << "," << r.center.y << ")" << std::endl;
        } else {
            std::cout << "FrameID: " << frameID << "; valid=0" << std::endl;
        }
        frameID++;
        drawResult(frame, r);
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
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    std::string color = argv[2];
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return 1;
    }
    std::string ext = std::filesystem::path(path).extension().string();
    if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
        processVideo(path, color);
    } else {
        processImage(path, color);
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
    } else {
        return false;
    }
    return true;
}

DetectResult detectColor(const cv::Mat& bgr, const std::string& color) {
    DetectResult r;
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
    r.valid = 1;
    r.area = maxArea;
    return r;
}

void drawResult(cv::Mat& img, const DetectResult& r) {
    if (r.valid) {
        cv::circle(img, cv::Point(static_cast<int>(r.center.x), static_cast<int>(r.center.y)), 6, cv::Scalar(0, 0, 255), -1);
        cv::putText(img, "valid=1", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    } else {
        cv::putText(img, "valid=0", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }
}
