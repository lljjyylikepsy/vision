#include "common.hpp"
#include <vector>
#include <string>

#ifdef HAVE_TESSERACT
#include <tesseract/baseapi.h>
#endif

struct DigitResult {
    int valid = 0;
    int label = -1;
    cv::Point2f center = {0, 0};
    double confidence = 0.0;
};

std::vector<DigitResult> detectDigits(const cv::Mat& img, double confMin);
void drawResults(cv::Mat& img, const std::vector<DigitResult>& results);

void processImage(const std::string& path) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    std::vector<DigitResult> results = detectDigits(img, 50.0);
    //
    double costMs = common::tickMs(t0);
    std::cout << "digits = " << results.size() << "; cost = " << costMs << "ms" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "[" << i << "] valid=" << results[i].valid << "; label=" << results[i].label << "; center=(" << results[i].center.x << "," << results[i].center.y << "); confidence=" << results[i].confidence << std::endl;
    }
    cv::Mat copy = img.clone();
    drawResults(copy, results);
    cv::imwrite(common::outputPath("task6_result.png").string(), copy);
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return 1;
    }
    processImage(path);
    return 0;
}

std::vector<DigitResult> detectDigits(const cv::Mat& img, double confMin) {
    std::vector<DigitResult> results;
    cv::Mat grey;
    cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(grey, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (binary.at<uchar>(0, 0) > 0) cv::bitwise_not(binary, binary);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hier;
    cv::findContours(binary.clone(), contours, hier, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<int> idx;
    for (size_t i = 0; i < contours.size(); ++i) {
        int parent = hier[i][3];
        if (parent >= 0 && hier[parent][3] == -1) idx.push_back(static_cast<int>(i));
    }
    if (idx.empty()) {
        for (size_t i = 0; i < contours.size(); ++i) {
            if (hier[i][3] == -1) idx.push_back(static_cast<int>(i));
        }
    }
#ifdef HAVE_TESSERACT
    tesseract::TessBaseAPI api;
    api.Init(NULL, "eng");
    api.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
    api.SetVariable("tessedit_char_whitelist", "0123456789");
#endif
    for (size_t k = 0; k < idx.size(); ++k) {
        int i = idx[k];
        if (cv::contourArea(contours[i]) < 100.0) continue;
        cv::Rect box = cv::boundingRect(contours[i]);
        if (box.width < 5 || box.height < 5) continue;
        DigitResult r;
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 != 0) {
            r.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
        }
#ifdef HAVE_TESSERACT
        cv::Mat roi = binary(box).clone();
        cv::Mat big;
        double scale = 100.0 / roi.rows;
        cv::resize(roi, big, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::Mat pad;
        cv::copyMakeBorder(big, pad, 20, 20, 20, 20, cv::BORDER_CONSTANT, cv::Scalar(255));
        api.SetImage(pad.data, pad.cols, pad.rows, 1, static_cast<int>(pad.step));
        char* txt = api.GetUTF8Text();
        int conf = api.MeanTextConf();
        api.Clear();
        std::string s = txt ? txt : "";
        delete[] txt;
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
        if (!s.empty() && s[0] >= '0' && s[0] <= '9') {
            r.label = s[0] - '0';
            r.confidence = conf < 0 ? 0 : conf;
            r.valid = r.confidence >= confMin ? 1 : 0;
        }
#endif
        results.push_back(r);
    }
    return results;
}

void drawResults(cv::Mat& img, const std::vector<DigitResult>& results) {
    for (size_t i = 0; i < results.size(); ++i) {
        cv::circle(img, cv::Point(static_cast<int>(results[i].center.x), static_cast<int>(results[i].center.y)), 5, cv::Scalar(0, 0, 255), -1);
        std::string text = cv::format("%d (%.0f)", results[i].label, results[i].confidence);
        cv::putText(img, text, cv::Point(static_cast<int>(results[i].center.x) + 8, static_cast<int>(results[i].center.y) - 8), cv::FONT_HERSHEY_SIMPLEX, 0.5, results[i].valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
    }
}
