#include "common.hpp"
#include <vector>

struct Params {
    int thresh = 127;
    int morphSize = 5;
    double minArea = 500.0;
};

cv::Mat cvting(const cv::Mat& img);
cv::Mat makeMask(const cv::Mat& img, const Params& p);
std::vector<std::vector<cv::Point>> findTargets(const cv::Mat& mask, const Params& p);
void drawOverlay(cv::Mat& img, const std::vector<std::vector<cv::Point>>& targets);

void processImage(const std::string& path, const Params& p) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    cv::Mat outcvt = cvting(img);
    cv::Mat mask = makeMask(img, p);
    auto targets = findTargets(mask, p);
    std::cout << "targets = " << targets.size() << std::endl;
    cv::Mat copy = img.clone();
    drawOverlay(copy, targets);
    //
    auto out = common::outputPath("task3_cvtResult.png");
    cv::imwrite(out.string(), outcvt);
    out = common::outputPath("task3_maskResult.png");
    cv::imwrite(out.string(), mask);
    out = common::outputPath("task3_overlayResult.png");
    cv::imwrite(out.string(), copy);
    //
    double costMs = common::tickMs(t0);
    std::cout << "cost " << costMs << "ms" << std::endl;
    std::cout << "saved cvt / mask / overlay results" << std::endl;
    cv::imshow("cvtResult", outcvt);
    cv::imshow("maskResult", mask);
    cv::imshow("overlayResult", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

int main (int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return 1;
    }
    Params p;
    if (argc >= 3) p.thresh = std::stoi(argv[2]);
    std::string ext = std::filesystem::path(path).extension().string();
    processImage(path, p);
    return 0;
}

cv::Mat cvting(const cv::Mat& img) {
    cv::Mat hsv, grey, grey3;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, grey3, cv::COLOR_GRAY2BGR);
    cv::Mat montage;
    cv::hconcat(std::vector<cv::Mat>{img, grey3, hsv}, montage);
    return montage;
}

cv::Mat makeMask(const cv::Mat& img, const Params& p){
    cv::Mat grey, binary;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::threshold(grey, binary, p.thresh, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(p.morphSize, p.morphSize));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    return binary;
}

std::vector<std::vector<cv::Point>> findTargets(const cv::Mat& mask, const Params& p) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> kept;
    for (const auto& c : contours) {
        if (cv::contourArea(c) >= p.minArea) kept.push_back(c);
    }
    return kept;
}

void drawOverlay(cv::Mat& img, const std::vector<std::vector<cv::Point>>& targets) {
    cv::Point center(img.cols / 2, img.rows / 2);
    cv::circle(img, center, 5, cv::Scalar(0, 0, 255), -1);
    cv::arrowedLine(img, center, cv::Point(center.x + 60, center.y), cv::Scalar(0, 0, 255), 2);
    cv::arrowedLine(img, center, cv::Point(center.x, center.y + 60), cv::Scalar(0, 255, 0), 2);
    for (const auto& c : targets) {
        cv::rectangle(img, cv::boundingRect(c), cv::Scalar(0, 255, 0), 2);
    }
}
