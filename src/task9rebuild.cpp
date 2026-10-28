#include "common.hpp"
#include <vector>
#include <cmath>

struct ArrowResult {
    bool valid = false;
    cv::Point2f center;
    cv::Point2f tip;
    double angleDeg = 0.0;
    double length = 0.0;
};

ArrowResult detectArrow(const cv::Mat& img);
void drawResult(cv::Mat& img, const ArrowResult& r);
void printResult(const ArrowResult& r, double costMs);

void processImage(const std::string& path) {
    cv::Mat img;
    if (!common::loadImage(path, img)) return;
    double t0 = common::tick();
    //
    ArrowResult r = detectArrow(img);
    //
    double costMs = common::tickMs(t0);
    cv::Mat copy = img.clone();
    drawResult(copy, r);
    printResult(r, costMs);
    auto out = common::outputPath("task9rebuild_result.png");
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
        ArrowResult r = detectArrow(frame);
        //
        double costMs = common::tickMs(t0);
        drawResult(frame, r);
        std::cout << "FrameID: " << frameID << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost = " << costMs << "ms; valid = " << r.valid << "; angle = " << r.angleDeg << "deg; length = " << r.length << "px" << std::endl;
        frameID++;
        if (writer.isOpened()) writer.write(frame);
        cv::imshow("result", frame);
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
    std::string path = argv[1];
    if (common::isCameraIndex(path)) {
        auto cap = common::openCamera(path);
        if (!cap.isOpened()) return 1;
        processStream(cap, "task9rebuild_camera.avi");
        return 0;
    }
    if (!common::ensurePathExists(path)) return 1;
    if (common::isVideoFile(path)) {
        auto cap = common::openVideo(path);
        processStream(cap, "task9rebuild_video.avi");
    } else {
        processImage(path);
    }
    return 0;
}

ArrowResult detectArrow(const cv::Mat& img) {
    ArrowResult r;
    return r;
}

void drawResult(cv::Mat& img, const ArrowResult& r) {
    return;
}

void printResult(const ArrowResult& r, double costMs) {
    return;
}
