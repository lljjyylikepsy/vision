#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

void photoget(const std::string& path) {
    cv::Mat reading = cv::imread(path);
    if (reading.empty()) {
        std::cerr << "image error: " << path << std::endl;
        return;
    }
    cv::Mat showing = reading;
    std::filesystem::create_directories("output");
    std::filesystem::path out = std::filesystem::path("output") / "result.png";
    cv::imwrite(out.string(), showing);
    std::cout << "saving to: " << out.string() << std::endl;
    cv::imshow("windows1", showing);
    cv::waitKey(5000);
    return;
}

void videoget(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << path << std::endl;
        return;
    }
    cv::Mat frame;
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalfps = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frameID = 0;
    std::cout << "Total frames is " << totalfps << std::endl;
    const double tickFreq = cv::getTickFrequency();
    cv::VideoWriter writer;
    cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (fps > 0 && sz.width > 0 && sz.height > 0) {
        std::filesystem::create_directories("output");
        std::filesystem::path outPath = std::filesystem::path("output") / "result_video.avi";
        writer.open(outPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz);
        if (!writer.isOpened()) {
            std::cerr << "[warn] VideoWriter open failed: " << outPath.string() << std::endl;
        }else {
            std::cout << "saving to: " << outPath.string() << std::endl;
        }
    }
    while (1) {
        double t0 = static_cast<double>(cv::getTickCount());
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("Video", frame);
        if (!writer.isOpened()) break;
        writer.write(frame);
        double costMs = (static_cast<double>(cv::getTickCount()) - t0)/ tickFreq * 1000.0;
        std::cout << "FrameID: " << frameID++ << "; ts = " << cap.get(cv::CAP_PROP_POS_MSEC) << "ms; cost "  << costMs << "ms" << std::endl;
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    return;
}

void cameraget(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return;
    }
    cv::Mat frame;
    cap >> frame;
    std::filesystem::create_directories("output");
    std::filesystem::path outPath = std::filesystem::path("output") / "camera_result.avi";
    cv::VideoWriter writer;
    writer.open(outPath.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, cv::Size(frame.cols, frame.rows));
    if (!writer.isOpened()) {
        std::cerr << "[warn] VideoWriter open failed: " << outPath.string() << std::endl;
    } else {
        std::cout << "saving to: " << outPath.string() << std::endl;
    }
    int frameID = 0;
    const double tickFreq = cv::getTickFrequency();
    while (1) {
        cap >> frame;
        if (frame.empty()) break;
        double t0 = static_cast<double>(cv::getTickCount());
        cv::imshow("camera", frame);
        if (!writer.isOpened()) break;
        writer.write(frame);
        double costMs = (static_cast<double>(cv::getTickCount()) - t0)/ tickFreq * 1000.0;
        std::cout << "FrameID: " << frameID++ <<"; cost "  << costMs << "ms" << std::endl;
        if (cv::waitKey(30) == 27) break;
    }
    writer.release();
    cap.release();
    return;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <image or video path>" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    bool numeric = !path.empty() && path.find_first_not_of("0123456789") == std::string::npos;
    if (numeric) {
        int idx = std::stoi(path);
        cv::VideoCapture cap(idx);
        if (!cap.isOpened()) {
            std::cerr << "cannot open camera " << idx << std::endl;
            return 1;
        }
        std::cout << "camera " << idx << " opened" << std::endl;
        cameraget(cap);
        return 0;
    }
    if (!std::filesystem::exists(path)) {
        std::cerr << "path does not exist:" << path << std::endl;
        return 1;
    }
    cv::Mat img = cv::imread(path);
    if (!img.empty()) {
        photoget(path);
    } else {
        videoget(path);
    }
    cv::destroyAllWindows();
    return 0;
}