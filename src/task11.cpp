#include "common.hpp"
#include "board.hpp"
#include "arrow.hpp"
#include "target.hpp"
#include "calib.hpp"
#include "uart_sender.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

struct Task11Config {
    BoardConfig board;
    TargetSpec target;
    std::string calibPath = "output/calib.yml";
    std::string targetPath = "config/target.yml";
    std::string serialPath = "config/serial.yml";
    double deadbandDeg = 2.0;
};

struct Task11FrameResult {
    BoardResult board;
    ArrowResult arrow;
    TargetResult target;
    double targetAngle = 0.0;
    double error = 0.0;
    double motorCmd = 0.0;
    bool valid = false;
};

struct Task11State {
    Task11Config cfg;
    UartSender sender;
    std::ofstream statusLog;
    std::filesystem::file_time_type targetMtime;
    unsigned frameID = 0;
};

bool loadTaskCalib(Task11State& state);
bool reloadTargetSpec(Task11State& state);
double normalizeAngle(double deg);
double targetAngleDeg(const cv::Point2f& center, const cv::Point2f& target);
Task11FrameResult processFrame(const cv::Mat& frame, Task11State& state);
void drawResult(cv::Mat& img, const Task11FrameResult& r);
void logStatus(Task11State& state, const Task11FrameResult& r);

void processImage(const std::string& path, Task11State& state) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    Task11FrameResult r = processFrame(img, state);
    cv::Mat copy = img.clone();
    //
    double costMs = common::tickMs(t0);
    std::cout << "cost = " << costMs << "ms" << std::endl;
    drawResult(copy, r);
    logStatus(state, r);
    auto out = common::outputPath("task11_result.png");
    cv::imwrite(out.string(), copy);
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("task11", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName, Task11State& state) {
    if (!cap.isOpened()) return;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::VideoWriter writer;
    if (sz.width > 0 && sz.height > 0) {
        auto outPath = common::outputPath(outName);
        writer.open(outPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
        if (writer.isOpened()) std::cout << "saving to: " << outPath.string() << std::endl;
    }
    cv::Mat frame;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        double t0 = common::tick();
        //
        Task11FrameResult r = processFrame(frame, state);
        cv::Mat copy = frame.clone();
        //
        double costMs = common::tickMs(t0);
        std::cout << "FrameID: " << state.frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms; valid = " << r.valid << "; error = " << r.error << "deg" << std::endl;
        state.frameID++;
        drawResult(copy, r);
        logStatus(state, r);
        if (writer.isOpened()) writer.write(copy);
        cv::imshow("task11", copy);
        if (cv::waitKey(1) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    Task11State state;
    if (!loadTaskCalib(state)) {
        std::cerr << "[error] cannot load calibration: " << state.cfg.calibPath << std::endl;
        return 1;
    }
    if (!loadTargetSpec(state.cfg.targetPath, state.cfg.target)) {
        std::cout << "[warn] cannot read target spec: " << state.cfg.targetPath << std::endl;
    }
    std::error_code ec;
    state.targetMtime = std::filesystem::last_write_time(state.cfg.targetPath, ec);
    state.sender.open(state.cfg.serialPath, "uart_task11.log");
    state.sender.start();
    auto logPath = common::outputPath("task11_status.csv");
    state.statusLog.open(logPath.string(), std::ios::app);
    if (state.statusLog.is_open()) {
        state.statusLog << "# angle convention: up=0, CCW positive\n";
        state.statusLog << "# time_ms,frame_id,board_complete,target_label,target_conf,target_x,target_y,arrow_cx,arrow_cy,arrow_angle,target_angle,error,motor_cmd,valid,online\n";
        state.statusLog.flush();
    }
    std::string path = argv[1];
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task11_camera.avi", state);
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task11_video.avi", state);
    } else {
        processImage(path, state);
    }
    return 0;
}

bool loadTaskCalib(Task11State& state) {
    cv::Mat K;
    cv::Mat dist;
    cv::Size size;
    if (!loadCalib(state.cfg.calibPath, K, dist, size)) return false;
    state.cfg.board.cameraMatrix = K;
    state.cfg.board.distCoeffs = dist;
    state.cfg.board.calibSize = size;
    return true;
}

bool reloadTargetSpec(Task11State& state) {
    std::error_code ec;
    auto t = std::filesystem::last_write_time(state.cfg.targetPath, ec);
    if (ec) return false;
    if (t == state.targetMtime) return false;
    TargetSpec next;
    if (!loadTargetSpec(state.cfg.targetPath, next)) return false;
    state.cfg.target = next;
    state.targetMtime = t;
    return true;
}

double normalizeAngle(double deg) {
    return std::atan2(std::sin(deg * CV_PI / 180.0), std::cos(deg * CV_PI / 180.0)) * 180.0 / CV_PI;
}

double targetAngleDeg(const cv::Point2f& center, const cv::Point2f& target) {
    return std::atan2(-(target.x - center.x), -(target.y - center.y)) * 180.0 / CV_PI;
}

Task11FrameResult processFrame(const cv::Mat& frame, Task11State& state) {
    Task11FrameResult r;
    if (reloadTargetSpec(state)) {
        std::cout << "[info] target reloaded: color=" << state.cfg.target.color << "; shape=" << state.cfg.target.shape << "; digit=" << state.cfg.target.digit << std::endl;
    }
    detectBoard(frame, state.cfg.board, r.board);
    r.arrow = detectArrow(frame);
    r.target = detectTarget(frame, state.cfg.target);
    if (r.arrow.valid && r.target.valid) {
        r.targetAngle = targetAngleDeg(r.arrow.center, r.target.center);
        r.error = normalizeAngle(r.targetAngle - r.arrow.angleDeg);
        r.motorCmd = r.error;
        if (std::fabs(r.error) < state.cfg.deadbandDeg) r.motorCmd = 0.0;
        r.valid = true;
    }
    RecognitionResult out;
    out.valid = r.valid;
    out.hasAngle = r.valid;
    out.target = r.target.center;
    out.angleDeg = r.motorCmd;
    out.label = r.target.label;
    state.sender.update(out);
    return r;
}

void drawResult(cv::Mat& img, const Task11FrameResult& r) {
    if (r.target.valid) {
        cv::circle(img, r.target.center, 8, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, r.target.label, cv::Point(static_cast<int>(r.target.center.x) + 10, static_cast<int>(r.target.center.y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    if (r.arrow.valid) {
        cv::circle(img, r.arrow.center, 8, cv::Scalar(0, 255, 255), 2);
        cv::arrowedLine(img, r.arrow.center, r.arrow.tip, cv::Scalar(0, 0, 255), 3);
    }
    if (!r.arrow.region.empty()) {
        std::vector<std::vector<cv::Point>> regionContours;
        cv::findContours(r.arrow.region.clone(), regionContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(img, regionContours, -1, cv::Scalar(255, 0, 255), 2);
    }
    if (r.valid) cv::line(img, r.arrow.center, r.target.center, cv::Scalar(255, 0, 0), 2);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "valid %d  board %d  arrow %.1f  target %.1f  error %.1f  cmd %.1f (CCW+)", r.valid, r.board.complete, r.arrow.angleDeg, r.targetAngle, r.error, r.motorCmd);
    cv::putText(img, buf, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, r.valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
    return;
}

void logStatus(Task11State& state, const Task11FrameResult& r) {
    if (!state.statusLog.is_open()) return;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    state.statusLog << ms << ',' << state.frameID << ',' << (r.board.complete ? 1 : 0) << ',' << r.target.label << ',' << r.target.confidence << ',' << r.target.center.x << ',' << r.target.center.y << ',' << r.arrow.center.x << ',' << r.arrow.center.y << ',' << r.arrow.angleDeg << ',' << r.targetAngle << ',' << r.error << ',' << r.motorCmd << ',' << (r.valid ? 1 : 0) << ',' << (state.sender.online() ? 1 : 0) << '\n';
    state.statusLog.flush();
}
