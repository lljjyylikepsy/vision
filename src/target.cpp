#include "target.hpp"
#include "color.hpp"
#include "shape.hpp"
#include "digit.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

bool loadTargetSpec(const std::string& path, TargetSpec& spec) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    if (!fs["color"].empty()) fs["color"] >> spec.color;
    if (!fs["shape"].empty()) fs["shape"] >> spec.shape;
    cv::FileNode digit = fs["digit"];
    if (!digit.empty()) {
        if (digit.isString()) {
            digit >> spec.digit;
        } else {
            int v = 0;
            digit >> v;
            spec.digit = std::to_string(v);
        }
    }
    fs.release();
    return true;
}

TargetResult detectTarget(const cv::Mat& img, const TargetSpec& spec) {
    TargetResult r;
    bool requiredOk = true;
    bool hasRef = false;
    cv::Point2f ref(0, 0);
    double conf = 1.0;
    std::vector<std::string> labels;
    if (!spec.color.empty()) {
        labels.push_back(spec.color);
        ColorResult c = detectColor(img, spec.color);
        if (c.valid) {
            r.center = c.center;
            ref = c.center;
            hasRef = true;
        } else {
            requiredOk = false;
        }
    }
    if (!spec.shape.empty()) {
        labels.push_back(spec.shape);
        std::vector<ShapeResult> shapes = detectShapes(img, 300.0);
        int best = -1;
        double bestScore = -1e18;
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (shapes[i].label != spec.shape) continue;
            double score = shapes[i].area;
            if (hasRef) score = -cv::norm(shapes[i].center - ref);
            if (score > bestScore) {
                bestScore = score;
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) {
            r.center = shapes[best].center;
            ref = r.center;
            hasRef = true;
        } else {
            requiredOk = false;
        }
    }
    if (!spec.digit.empty()) {
        labels.push_back(spec.digit);
        std::vector<DigitResult> digits = detectDigits(img, 0.5);
        int want = -1;
        try {
            want = std::stoi(spec.digit);
        } catch (...) {
            return r;
        }
        int best = -1;
        double bestScore = -1e18;
        for (size_t i = 0; i < digits.size(); ++i) {
            if (!digits[i].valid || digits[i].label != want) continue;
            double score = digits[i].confidence;
            if (hasRef) score = -cv::norm(digits[i].center - ref);
            if (score > bestScore) {
                bestScore = score;
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) {
            r.center = digits[best].center;
            ref = r.center;
            hasRef = true;
            conf = std::min(conf, digits[best].confidence);
        } else {
            requiredOk = false;
        }
    }
    if (labels.empty() || !requiredOk || !hasRef) return r;
    if (conf < 0.0) conf = 0.0;
    if (conf > 1.0) conf = 1.0;
    r.valid = true;
    r.confidence = conf;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i) r.label += "_";
        r.label += labels[i];
    }
    return r;
}
