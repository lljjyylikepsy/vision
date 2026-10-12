#pragma once                      // 防止同一个头文件被重复包含

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>

// =============================================================================
// common.hpp —— 各任务共用的通用工具
// 只放“不依赖具体任务”的东西：计时、输出目录、读图、打开视频。
// 注意：头文件里定义的函数必须加 inline，否则被多个 .cpp include 会重定义。
// =============================================================================

namespace common {

// -----------------------------------------------------------------------------
// 1. 计时：OpenCV 自带 tick 计时，返回毫秒
//    用法：double t0 = common::tick();  ...  double ms = common::tickMs(t0);
// -----------------------------------------------------------------------------
inline double tick() {
    return static_cast<double>(cv::getTickCount());
}

inline double tickMs(double t0) {
    return (static_cast<double>(cv::getTickCount()) - t0) / cv::getTickFrequency() * 1000.0;
}

// -----------------------------------------------------------------------------
// 2. 输出目录：确保 output/ 存在，并返回其中某个文件的完整路径
//    用法：auto p = common::outputPath("result.png");  // output/result.png
// -----------------------------------------------------------------------------
inline std::filesystem::path outputPath(const std::string& filename) {
    std::filesystem::create_directories("output");        // 多级目录也能建
    return std::filesystem::path("output") / filename;
}

// -----------------------------------------------------------------------------
// 3. 读图：带错误处理。成功返回 true，并把结果写入 img。
// -----------------------------------------------------------------------------
inline bool loadImage(const std::string& path, cv::Mat& img) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "[error] path not exist: " << path << std::endl;
        return false;
    }
    img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "[error] corrupted image: " << path << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// 4. 打开视频：带错误处理。返回已打开的 VideoCapture（失败时 isOpened() 为 false）。
// -----------------------------------------------------------------------------
inline cv::VideoCapture openVideo(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "[error] cannot open video: " << path << std::endl;
    }
    return cap;
}

} // namespace common
