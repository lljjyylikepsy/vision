#include "digit.hpp"
#include <string>
#ifdef HAVE_TESSERACT
#include <tesseract/baseapi.h>
#endif

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
    static tesseract::TessBaseAPI api;
    static bool inited = false;
    if (!inited) {
        api.Init(NULL, "eng");
        api.SetPageSegMode(tesseract::PSM_SINGLE_CHAR);
        api.SetVariable("tessedit_char_whitelist", "0123456789");
        inited = true;
    }
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
