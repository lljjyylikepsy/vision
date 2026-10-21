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
// 3. 路径检查：路径不存在时打印统一错误信息。
//    用法：if (!common::ensurePathExists(path)) return 1;
// -----------------------------------------------------------------------------
inline bool ensurePathExists(const std::string& path) {
    if (std::filesystem::exists(path)) return true;
    std::cerr << "[error] path not exist: " << path << std::endl;
    return false;
}

// -----------------------------------------------------------------------------
// 4. 输入源判断：纯数字视为摄像头索引；按扩展名识别常见视频格式。
//    用法：if (common::isCameraIndex(path)) { ... }
// -----------------------------------------------------------------------------
inline bool isCameraIndex(const std::string& path) {
    return !path.empty() && path.find_first_not_of("0123456789") == std::string::npos;
}

inline bool isVideoFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv";
}

// -----------------------------------------------------------------------------
// 5. 读图：带错误处理。成功返回 true，并把结果写入 img。
// -----------------------------------------------------------------------------
inline bool loadImage(const std::string& path, cv::Mat& img) {
    if (!ensurePathExists(path)) return false;
    img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "[error] corrupted image: " << path << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// 6. 打开摄像头/视频：带错误处理。返回已打开的 VideoCapture（失败时 isOpened() 为 false）。
// -----------------------------------------------------------------------------
inline cv::VideoCapture openCamera(const std::string& index) {
    cv::VideoCapture cap(std::stoi(index));
    if (!cap.isOpened()) {
        std::cerr << "[error] cannot open camera " << index << std::endl;
    }
    return cap;
}

inline cv::VideoCapture openVideo(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "[error] cannot open video: " << path << std::endl;
    }
    return cap;
}

} // namespace common
