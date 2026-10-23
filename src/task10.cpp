#include "common.hpp"
#include <string>
#ifdef HAVE_MOTOR
#include "control.h"
#endif

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

bool loadSerialConfig(const std::string& path, SerialConfig& cfg);
bool openSerial(const std::string& path);
UartFrame makeFrame(const RecognitionResult& r, unsigned seq);
#ifdef HAVE_MOTOR
control::Frame toControlFrame(const UartFrame& f);
#endif

void processImage(const std::string& path) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    cv::Mat copy = img.clone();
    //
    double costMs = common::tickMs(t0);
    std::cout << "cost = " << costMs << "ms" << std::endl;
    auto out = common::outputPath("task10_result.png");
    cv::imwrite(out.string(), copy);
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("result", copy);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void processStream(cv::VideoCapture& cap, const std::string& outName) {
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
    int frameID = 0;
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        double t0 = common::tick();
        //
        cv::Mat copy = frame.clone();
        //
        double costMs = common::tickMs(t0);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms" << std::endl;
        frameID++;
        if (writer.isOpened()) writer.write(copy);
        cv::imshow("result", copy);
        if (cv::waitKey(30) == 27) break;
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
    std::string cfgPath = "config/serial.yml";
    SerialConfig cfg;
    if (loadSerialConfig(cfgPath, cfg)) {
        std::cout << "[info] serial config: device=" << cfg.device << "; baud=" << cfg.baud << "; period=" << cfg.periodMs << "ms; protocol=" << cfg.protocol << "; format=" << cfg.textFormat << "; stop=" << cfg.stopValue << std::endl;
    } else {
        std::cout << "[warn] cannot read " << cfgPath << "; using defaults" << std::endl;
    }
    openSerial(cfgPath);
    std::string path = argv[1];
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task10_camera.avi");
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task10_video.avi");
    } else {
        processImage(path);
    }
    return 0;
}

bool loadSerialConfig(const std::string& path, SerialConfig& cfg) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    if (!fs["device"].empty()) fs["device"] >> cfg.device;
    if (!fs["baud"].empty()) fs["baud"] >> cfg.baud;
    if (!fs["period_ms"].empty()) fs["period_ms"] >> cfg.periodMs;
    if (!fs["protocol"].empty()) fs["protocol"] >> cfg.protocol;
    if (!fs["text_format"].empty()) fs["text_format"] >> cfg.textFormat;
    if (!fs["stop_value"].empty()) fs["stop_value"] >> cfg.stopValue;
    fs.release();
    return true;
}

bool openSerial(const std::string& path) {
#ifdef HAVE_MOTOR
    bool ok = control::open(path);
    std::cout << "[info] serial " << (control::online() ? "online" : "offline") << "; period = " << control::periodMs() << "ms" << std::endl;
    return ok;
#else
    (void)path;
    std::cout << "[warn] built without motor module; serial disabled" << std::endl;
    return false;
#endif
}

UartFrame makeFrame(const RecognitionResult& r, unsigned seq) {
    UartFrame f;
    f.seq = seq;
    f.valid = r.valid;
    f.hasAngle = r.valid && r.hasAngle;
    f.x = r.target.x;
    f.y = r.target.y;
    f.angle = r.angleDeg;
    f.label = r.label;
    return f;
}

#ifdef HAVE_MOTOR
control::Frame toControlFrame(const UartFrame& f) {
    control::Frame out;
    out.seq = f.seq;
    out.valid = f.valid;
    out.hasAngle = f.hasAngle;
    out.x = f.x;
    out.y = f.y;
    out.angle = f.angle;
    out.label = f.label;
    return out;
}
#endif
