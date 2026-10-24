#pragma once
#include "common.hpp"
#include <string>
#include <chrono>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>

struct RecognitionResult {
    bool valid = false;
    bool hasAngle = false;
    cv::Point2f target{0.0f, 0.0f};
    double angleDeg = 0.0;
    std::string label;
};

struct SerialConfig {
    std::string device = "/dev/ttyUSB0";
    int baud = 115200;
    int periodMs = 30;
    std::string protocol = "text";
    std::string textFormat = "%.0f";
    double stopValue = 999.0;
};

struct UartFrame {
    unsigned seq = 0;
    bool valid = false;
    bool hasAngle = false;
    float x = 0.0f;
    float y = 0.0f;
    double angle = 0.0;
    std::string label;
};

struct UartSender {
    SerialConfig cfg;
    unsigned seq = 0;
    std::chrono::steady_clock::time_point next;
    std::ofstream log;
    std::thread worker;
    std::atomic<bool> running{false};
    std::mutex mtx;
    RecognitionResult latest;
    std::chrono::steady_clock::time_point lastUpdate;
    ~UartSender();
    bool open(const std::string& cfgPath, const std::string& logName = "uart_task10.log");
    void start();
    void stop();
    void update(const RecognitionResult& r);
    void loop();
    void waitNext();
    bool send(const RecognitionResult& r, bool turn);
    bool online() const;
    void logLine(const UartFrame& f, bool turn, bool sent);
};

bool loadSerialConfig(const std::string& path, SerialConfig& cfg);
bool openSerial(const std::string& path);
UartFrame makeFrame(const RecognitionResult& r, unsigned seq);
