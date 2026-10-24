#pragma once
#include "common.hpp"

bool loadCalib(const std::string& path, cv::Mat& K, cv::Mat& dist, cv::Size& size);
