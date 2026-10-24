#include "common.hpp"
#include <string>
#include <chrono>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
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
    bool open(const std::string& cfgPath);
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
#ifdef HAVE_MOTOR
control::Frame toControlFrame(const UartFrame& f);
#endif

void processImage(const std::string& path, UartSender& sender, const RecognitionResult& base) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    RecognitionResult r = base;
    r.target = cv::Point2f(img.cols / 2.0f, img.rows / 2.0f);
    sender.update(r);
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

void processStream(cv::VideoCapture& cap, const std::string& outName, UartSender& sender, const RecognitionResult& base) {
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
        RecognitionResult r = base;
        r.target = cv::Point2f(frame.cols / 2.0f, frame.rows / 2.0f);
        sender.update(r);
        cv::Mat copy = frame.clone();
        //
        double costMs = common::tickMs(t0);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms" << std::endl;
        frameID++;
        if (writer.isOpened()) writer.write(copy);
        cv::imshow("result", copy);
        if (cv::waitKey(1) == 27) break;
    }
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path> [testAngleDeg]" << std::endl;
        return 1;
    }
    UartSender sender;
    sender.open("config/serial.yml");
    RecognitionResult base;
    if (argc >= 3) {
        base.valid = true;
        base.hasAngle = true;
        base.angleDeg = std::stod(argv[2]);
        base.label = "arrow";
    } else {
        base.label = "none";
    }
    sender.start();
    std::string path = argv[1];
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task10_camera.avi", sender, base);
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task10_video.avi", sender, base);
    } else {
        processImage(path, sender, base);
    }
    return 0;
}

UartSender::~UartSender() {
    stop();
}

void UartSender::start() {
    if (running.load()) return;
    next = std::chrono::steady_clock::now();
    lastUpdate = next;
    running.store(true);
    worker = std::thread(&UartSender::loop, this);
}

void UartSender::stop() {
    if (!running.exchange(false)) return;
    if (worker.joinable()) worker.join();
}

void UartSender::update(const RecognitionResult& r) {
    std::lock_guard<std::mutex> lock(mtx);
    latest = r;
    lastUpdate = std::chrono::steady_clock::now();
}

void UartSender::loop() {
    while (running.load()) {
        waitNext();
        RecognitionResult r;
        {
            std::lock_guard<std::mutex> lock(mtx);
            r = latest;
            auto now = std::chrono::steady_clock::now();
            if (now - lastUpdate > std::chrono::milliseconds(200)) {
                r.valid = false;
                r.hasAngle = false;
            }
        }
        send(r, r.valid && r.hasAngle);
    }
}

bool UartSender::open(const std::string& cfgPath) {
    if (loadSerialConfig(cfgPath, cfg)) {
        std::cout << "[info] serial config: device=" << cfg.device << "; baud=" << cfg.baud << "; period=" << cfg.periodMs << "ms; protocol=" << cfg.protocol << "; format=" << cfg.textFormat << "; stop=" << cfg.stopValue << std::endl;
    } else {
        std::cout << "[warn] cannot read " << cfgPath << "; using defaults" << std::endl;
    }
    bool ok = openSerial(cfgPath);
    next = std::chrono::steady_clock::now();
    auto out = common::outputPath("uart_task10.log");
    log.open(out.string(), std::ios::app);
    if (log.is_open()) {
        log << "# time_ms,seq,valid,hasAngle,x,y,angle,turn,sent,online,label\n";
        log.flush();
    }
    return ok;
}

void UartSender::waitNext() {
    auto now = std::chrono::steady_clock::now();
    if (next < now) next = now;
    std::this_thread::sleep_until(next);
    int period = cfg.periodMs > 0 ? cfg.periodMs : 30;
    next += std::chrono::milliseconds(period);
    now = std::chrono::steady_clock::now();
    if (next < now) next = now + std::chrono::milliseconds(period);
}

bool UartSender::send(const RecognitionResult& r, bool turn) {
    UartFrame f = makeFrame(r, seq);
    bool sent = false;
#ifdef HAVE_MOTOR
    control::Frame cf = toControlFrame(f);
    if (turn && f.valid && f.hasAngle) sent = control::TurnAngle(cf, f.angle);
    else sent = control::Send(cf);
#else
    (void)turn;
#endif
    logLine(f, turn, sent);
    seq++;
    if (seq == 0) seq = 1;
    return sent;
}

bool UartSender::online() const {
#ifdef HAVE_MOTOR
    return control::online();
#else
    return false;
#endif
}

void UartSender::logLine(const UartFrame& f, bool turn, bool sent) {
    if (!log.is_open()) return;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    log << ms << ',' << f.seq << ',' << (f.valid ? 1 : 0) << ',' << (f.hasAngle ? 1 : 0) << ',' << f.x << ',' << f.y << ',' << f.angle << ',' << (turn ? 1 : 0) << ',' << (sent ? 1 : 0) << ',' << (online() ? 1 : 0) << ',' << f.label << '\n';
    log.flush();
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
