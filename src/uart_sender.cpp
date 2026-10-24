#include "uart_sender.hpp"
#ifdef HAVE_MOTOR
#include "control.h"
#endif

namespace {

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

bool UartSender::open(const std::string& cfgPath, const std::string& logName) {
    if (loadSerialConfig(cfgPath, cfg)) {
        std::cout << "[info] serial config: device=" << cfg.device << "; baud=" << cfg.baud << "; period=" << cfg.periodMs << "ms; protocol=" << cfg.protocol << "; format=" << cfg.textFormat << "; stop=" << cfg.stopValue << std::endl;
    } else {
        std::cout << "[warn] cannot read " << cfgPath << "; using defaults" << std::endl;
    }
    bool ok = openSerial(cfgPath);
    next = std::chrono::steady_clock::now();
    auto out = common::outputPath(logName);
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
