#include "calib.hpp"

bool loadCalib(const std::string& path, cv::Mat& K, cv::Mat& dist, cv::Size& size) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    int w = 0;
    int h = 0;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    size = cv::Size(w, h);
    fs.release();
    return !K.empty();
}
